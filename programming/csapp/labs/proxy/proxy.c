#include <stdio.h>
#include <assert.h>

#include "csapp.h"

/* Recommended max cache and object sizes */
#define MAX_CACHE_SIZE 1049000
#define MAX_OBJECT_SIZE 102400

/* You won't lose style points for including this long line in your code */
static const char *user_agent_hdr = "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:10.0.3) Gecko/20120305 Firefox/10.0.3\r\n";

void check_one_line(char const *line, ssize_t n) {
    assert(n >= 2);
    assert(line[n - 2] == '\r');
    assert(line[n - 1] == '\n');
}

int read_one_line(rio_t *rio, char *line) {
    ssize_t n = Rio_readlineb(rio, line, MAXLINE);
    check_one_line(line, n);
    printf("[C >> P] %s", line);
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

void forward_request(int server_fd, char const *method, char const *uri,
        char const *hostname, char const *buf) {
    char line[MAXLINE];

    // Send the request line:
    sprintf(line, "%s %s HTTP/1.0\r\n", method, uri);
    printf("[P >> S] %s", line);
    Rio_writen(server_fd, line, strlen(line));
    // Send the HOST header:
    sprintf(line, "Host: %s\r\n", hostname);
    printf("[P >> S] %s", line);
    Rio_writen(server_fd, line, strlen(line));
    // Send the User-Agent header:
    printf("[P >> S] %s", user_agent_hdr);
    Rio_writen(server_fd, (void *)user_agent_hdr, strlen(user_agent_hdr));
    // Send the Connection header:
    sprintf(line, "%s\r\n", "Connection: close");
    printf("[P >> S] %s", line);
    Rio_writen(server_fd, line, strlen(line));
    // Send the Connection header:
    sprintf(line, "%s\r\n", "Proxy-Connection: close");
    printf("[P >> S] %s", line);
    Rio_writen(server_fd, line, strlen(line));
    // Send other headers:
    char *ptr;
    do {
        ptr = strstr(buf, "\r\n");
        size_t len = ptr - buf + 2;
        strncpy(line, buf, len);
        line[len] = '\0';
        if (has_prefix(line, "GET") || has_prefix(line, "Host") ||
                has_prefix(line, "User-Agent") ||
                has_prefix(line, "Connection") ||
                has_prefix(line, "Proxy-Connection")) {
            printf("[ignore] %s", line);
        } else {
            printf("[P >> S] %s", line);
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

void forward_response(rio_t *server_rio, int server_fd,
        int client_fd, char *buf) {
    ssize_t n;
    // forward the headers
    do {
      n = Rio_readlineb(server_rio, buf, MAXLINE);
      check_one_line(buf, n);
      printf("[S >> C] %s", buf);
      Rio_writen(client_fd, buf, n);
    } while (n != 2);
    // forward the HTML
    while ((n = Rio_readlineb(server_rio, buf, MAXLINE))) {
      printf("[S >> C] %s", buf);
      Rio_writen(client_fd, buf, n);
    }
}

void serve(int client_fd) {
    // Read the entire HTTP request from the client and check whether the it is valid.
    size_t n = 0;
    char buf[MAXLINE], *line = buf;
    rio_t client_rio; Rio_readinitb(&client_rio, client_fd);
    // Read the first line:
    char method[MAXLINE], uri_from_client[MAXLINE], version[MAXLINE];
    n = read_one_line(&client_rio, line);
    sscanf(line, "%s %s %s", method, uri_from_client, version);
    if (strcasecmp(method, "GET")) {
        clienterror(client_fd, method, "501", "Not Implemented",
                    "Tiny Proxy does not implement this method");
        return;
    }
    printf("* method = \"%s\"\n", method);
    printf("* version = \"%s\"\n", version);
    printf("* uri_from_client = \"%s\"\n", uri_from_client);
    // Parse the URI from the client
    char hostname[MAXLINE];
    char const *uri_to_server = parse_uri(uri_from_client, hostname);
    printf("  * hostname = \"%s\"\n", hostname);
    printf("  * uri_to_server = \"%s\"\n", uri_to_server);
    // Read other lines:
    do {
      line += n;  // n does not account '\0'
      n = read_one_line(&client_rio, line);
    } while (n != 2);
    assert(strcmp(line, "\r\n") == 0);
    assert(strlen(buf) <= MAXLINE);
    printf("Length of the request: %ld\n", strlen(buf));
    // Connect to the appropriate web server.
    int server_fd;
    char *port = strchr(hostname, ':');
    if (port) {  // port explicitly given, use it
      *port++ = '\0';
    }
    server_fd = Open_clientfd(hostname, port ? port : "80");
    if (port) {  // port explicitly given, use it
      *--port = ':';
    }
    printf("Connected to (%s)\n", hostname);
    // Request the object the client specified.
    rio_t server_rio; Rio_readinitb(&server_rio, server_fd);
    forward_request(server_fd, method, uri_to_server, hostname, buf);
    // Read the server's response and forward it to the client.
    printf("Forward response from server (%s) to client\n", hostname);
    forward_response(&server_rio, server_fd, client_fd, buf);
    Close(server_fd);
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

    int listen_fd = Open_listenfd(argv[1]);
    while (1) {
        int client_fd = Accept(listen_fd, (SA *)&client_addr, &client_len);
        Getnameinfo((SA *) &client_addr, client_len,
            client_host, MAXLINE, client_port, MAXLINE, 0);
        printf("Connected to (%s, %s)\n", client_host, client_port);
        serve(client_fd);
        Close(client_fd);
    }
    exit(0);
}
