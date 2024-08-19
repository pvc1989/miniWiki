#include <stdio.h>
#include <assert.h>

#include "csapp.h"
#include "lru.h"  // the LRU cache

/* Recommended max cache and object sizes */
#define MAX_CACHE_SIZE 1049000
#define MAX_OBJECT_SIZE 102400

static lru_t *lru = NULL;

/* You won't lose style points for including this long line in your code */
static const char *user_agent_hdr = "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:10.0.3) Gecko/20120305 Firefox/10.0.3\r\n";

#define CONCURRENT

#ifndef CONCURRENT
#define PRINTF(...) (printf(__VA_ARGS__))
#define LRU_PRINT(...) (lru_print(__VA_ARGS__))
#define PTHREAD_PRINTF(...) (printf(__VA_ARGS__))
#else
static pthread_rwlock_t lru_rwlock = PTHREAD_RWLOCK_INITIALIZER;
#define PRINTF(...) ((void)0)
#define LRU_PRINT(...) ((void)0)

static pthread_rwlock_t stdout_rwlock = PTHREAD_RWLOCK_INITIALIZER;
#define PTHREAD_PRINTF(...) \
    pthread_rwlock_wrlock(&stdout_rwlock); \
    printf("[ThreadID = %ld] ", pthread_self()); \
    printf(__VA_ARGS__); \
    pthread_rwlock_unlock(&stdout_rwlock);
#endif

void check_one_line(char const *line, ssize_t n) {
    if (n < 2) {
        PRINTF("%s\n", line);
        return;
    }
    assert(line[n - 2] == '\r');
    assert(line[n - 1] == '\n');
}

int read_one_line(rio_t *rio, char const *side, char *line) {
    ssize_t n = Rio_readlineb(rio, line, MAXLINE);
    check_one_line(line, n);
    PRINTF("[%s >> P] %s", side, line);
    return n;
}

/**
 * @brief Returns an error message to the client.
 * 
 * Borrowed from `tiny.c`.
 * 
 */
void clienterror(int fd, char *cause, char *errnum, 
                 char *shortmsg, char *longmsg)
{
    char buf[MAXLINE];

    /* Print the HTTP response headers */
    sprintf(buf, "HTTP/1.0 %s %s\r\n", errnum, shortmsg);
    Rio_writen(fd, buf, strlen(buf));
    sprintf(buf, "Content-type: text/html\r\n\r\n");
    Rio_writen(fd, buf, strlen(buf));

    /* Print the HTTP response body */
    sprintf(buf, "<html><title>Tiny Proxy Error</title>");
    Rio_writen(fd, buf, strlen(buf));
    sprintf(buf, "<body bgcolor=""ffffff"">\r\n");
    Rio_writen(fd, buf, strlen(buf));
    sprintf(buf, "%s: %s\r\n", errnum, shortmsg);
    Rio_writen(fd, buf, strlen(buf));
    sprintf(buf, "<p>%s: %s\r\n", longmsg, cause);
    Rio_writen(fd, buf, strlen(buf));
    sprintf(buf, "<hr><em>The Tiny Proxy server</em>\r\n");
    Rio_writen(fd, buf, strlen(buf));
}

/**
 * @brief Parse the URI from the client into the hostname and the URI to the server.
 * @return the address of the first char in the URI to the server
 */
char const *parse_uri(char const *uri_from_client, char *hostname) 
{
    char const *ptr, *uri_to_server;

    assert(strncmp(uri_from_client, "http", 4) == 0);

    /* `strchr(str, ch)` finds the first occurrence of `ch` in `str` */
    ptr = strchr(uri_from_client, '/');
    ptr += 2;
    uri_to_server = strchr(ptr, '/');
    size_t len = uri_to_server - ptr;
    strncpy(hostname, ptr, len);
    hostname[len] = '\0';

    return uri_to_server;
}

size_t min(size_t x, size_t y) {
  return x < y ? x : y;
}

int has_prefix(char const *line, char const *prefix) {
    return !strncmp(line, prefix, min(strlen(line), strlen(prefix)));
}

int connect_to_server(char *hostname) {
    // Connect to the appropriate web server.
    int server_fd;
    char *port = strchr(hostname, ':');
    if (port) {  // port explicitly given, use it
      *port++ = '\0';
    }
    server_fd = Open_clientfd(hostname, port ? port : "80");
    if (port) {  // hostname modified, recover it
      *--port = ':';
    }
    PTHREAD_PRINTF("Connected to server %s\n", hostname);
    return server_fd;
}

void forward_request(int server_fd, char const *method, char const *uri,
        char const *hostname, char const *buf) {
    char line[MAXLINE];

    // Send the request line:
    sprintf(line, "%s %s HTTP/1.0\r\n", method, uri);
    PRINTF("[P >> S] %s", line);
    Rio_writen(server_fd, line, strlen(line));
    // Send the HOST header:
    sprintf(line, "Host: %s\r\n", hostname);
    PRINTF("[P >> S] %s", line);
    Rio_writen(server_fd, line, strlen(line));
    // Send the User-Agent header:
    PRINTF("[P >> S] %s", user_agent_hdr);
    Rio_writen(server_fd, (void *)user_agent_hdr, strlen(user_agent_hdr));
    // Send the Connection header:
    sprintf(line, "%s\r\n", "Connection: close");
    PRINTF("[P >> S] %s", line);
    Rio_writen(server_fd, line, strlen(line));
    // Send the Connection header:
    sprintf(line, "%s\r\n", "Proxy-Connection: close");
    PRINTF("[P >> S] %s", line);
    Rio_writen(server_fd, line, strlen(line));
    // Send other headers:
    char const *ptr;
    do {
        ptr = strstr(buf, "\r\n");
        size_t len = ptr - buf + 2;
        strncpy(line, buf, len);
        line[len] = '\0';
        if (has_prefix(line, "GET") || has_prefix(line, "Host") ||
                has_prefix(line, "User-Agent") ||
                has_prefix(line, "Connection") ||
                has_prefix(line, "Proxy-Connection")) {
            PRINTF("[ignore] %s", line);
        } else {
            PRINTF("[P >> S] %s", line);
            Rio_writen(server_fd, line, strlen(line));
        }
        if (buf == ptr) {
            break;
        }
        buf = ptr + 2;
    } while (1);
    assert(strcmp(line, "\r\n") == 0);
    assert(strcmp(buf, "\r\n") == 0);
    assert(buf == ptr);
}

void forward_response(int server_fd, int client_fd,
        char const *uri_from_client, char *caller_buf) {
    rio_t server_rio; Rio_readinitb(&server_rio, server_fd);
    ssize_t size = 0;
    /* Read at most MAX_OBJECT_SIZE bytes of the response. */
    char buf[MAX_OBJECT_SIZE];
    ssize_t n  // #bytes read in one call of Rio_readnb()
        = Rio_readnb(&server_rio, buf, MAX_OBJECT_SIZE);
    if (n == -1) {
        PRINTF("Error in Rio_readnb()\n");
        exit(-1);
    }
    size += n;
    // Send the part just read:
    Rio_writen(client_fd, buf, size);
    while ((n = Rio_readnb(&server_rio, caller_buf, MAXLINE)) != 0) {
        // Large object, no cache, read and send the remaing part:
        if (n == -1) {
            PRINTF("Error in Rio_readnb()\n");
            exit(-1);
        }
        assert(n > 0);
        size += n;
        Rio_writen(client_fd, caller_buf, n);
    }
    PRINTF("size = %ld\n", size);
    if (size > MAX_CACHE_SIZE) {
        return;
    }
    pthread_rwlock_wrlock(&lru_rwlock);
    // The item might already been emplaced by another thread, so find it again:
    if (!lru_find(lru, uri_from_client)) {
        lru_emplace(lru, uri_from_client, buf, size);
    }
    pthread_rwlock_unlock(&lru_rwlock);
    PTHREAD_PRINTF("Cache updated.\n");
}

void serve(int client_fd) {
    // Read the entire HTTP request from the client and check whether the it is valid.
    size_t n = 0;
    char buf[MAXLINE], *line = buf;
    rio_t client_rio; Rio_readinitb(&client_rio, client_fd);
    // Read the first line:
    char method[MAXLINE], uri_from_client[MAXLINE], version[MAXLINE];
    n = read_one_line(&client_rio, "C", line);
    sscanf(line, "%s %s %s", method, uri_from_client, version);
    if (strcasecmp(method, "GET")) {
        clienterror(client_fd, method, "501", "Not Implemented",
                    "Tiny Proxy does not implement this method");
        return;
    }
    PRINTF("* method = \"%s\"\n", method);
    PRINTF("* version = \"%s\"\n", version);
    PRINTF("* uri_from_client = \"%s\"\n", uri_from_client);
    // Already cached?
    /* lru_find() is a read-only operation, so it only need a read-lock */
    pthread_rwlock_rdlock(&lru_rwlock);
    item_t *item = lru_find(lru, uri_from_client);
    if (item) {
        PTHREAD_PRINTF("Cache hit!\n");
        PRINTF("[cache] Before sinking:\n");
        LRU_PRINT(lru);
        Rio_writen(client_fd,
            (void *)item_data(item), item_size(item));
    } else {
        PTHREAD_PRINTF("Cache miss!\n");
    }
    pthread_rwlock_unlock(&lru_rwlock);
    if (item) {  /* lru_sink(), which is a writing operation,
        is called only if the item is found (cache hit) */
        pthread_rwlock_wrlock(&lru_rwlock);
        // The item might already been popped by another thread, so find it again:
        if ((item = lru_find(lru, uri_from_client)) != NULL) {
            lru_sink(lru, item);
            PRINTF("[cache] After sinking:\n");
            LRU_PRINT(lru);
        }
        pthread_rwlock_unlock(&lru_rwlock);
        Close(client_fd);
        return;
    }
    // Parse the URI from the client
    char hostname[MAXLINE];
    char const *uri_to_server = parse_uri(uri_from_client, hostname);
    PRINTF("  * hostname = \"%s\"\n", hostname);
    PRINTF("  * uri_to_server = \"%s\"\n", uri_to_server);
    // Read other lines:
    do {
      line += n;  // n does not account '\0'
      n = read_one_line(&client_rio, "C", line);
    } while (n != 2);
    assert(strcmp(line, "\r\n") == 0);
    assert(strlen(buf) <= MAXLINE);
    PRINTF("Length of the request: %ld\n", strlen(buf));
    // Request the object the client specified.
    int server_fd = connect_to_server(hostname);
    forward_request(server_fd, method, uri_to_server, hostname, buf);
    // Read the server's response and forward it to the client.
    PRINTF("Forward response from server (%s) to client\n", hostname);
    forward_response(server_fd, client_fd, uri_from_client, buf);
    Close(server_fd);
    Close(client_fd);
}

/**
 * @brief The routine run in a thread.
 */
void *routine(void *vargp) {
    int client_fd = *((int *)vargp);
    Pthread_detach(Pthread_self());
    Free(vargp);
    serve(client_fd);
    return NULL;
}

void serve_by_thread(int client_fd) {
    int *client_fd_ptr = Malloc(sizeof(int));
    *client_fd_ptr = client_fd;

    pthread_t tid;
    Pthread_create(&tid, NULL, routine, client_fd_ptr);
}

int main(int argc, char **argv)
{
    struct sockaddr_storage client_addr;  /* Enough space for any address */
    socklen_t client_len = sizeof(struct sockaddr_storage);
    char client_host[MAXLINE], client_port[MAXLINE];

    if (argc != 2) {
        fprintf(stderr, "usage: %s <port>\n", argv[0]);
        exit(0);
    }

    lru = lru_construct(MAX_CACHE_SIZE);
    int listen_fd = Open_listenfd(argv[1]);
    while (1) {
        int client_fd = Accept(listen_fd, (SA *)&client_addr, &client_len);
        Getnameinfo((SA *) &client_addr, client_len,
            client_host, MAXLINE, client_port, MAXLINE, 0);
        PTHREAD_PRINTF("Connected to client %s:%s\n", client_host, client_port);
#ifdef CONCURRENT
        serve_by_thread(client_fd);
#else
        serve(client_fd);
#endif
    }

    lru_destruct(lru);
    pthread_rwlock_destroy(&lru_rwlock);
    pthread_rwlock_destroy(&stdout_rwlock);
    exit(0);
}
