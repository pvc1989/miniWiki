#include <stdio.h>
#include <assert.h>

#include "csapp.h"

/* Recommended max cache and object sizes */
#define MAX_CACHE_SIZE 1049000
#define MAX_OBJECT_SIZE 102400

/* You won't lose style points for including this long line in your code */
static const char *user_agent_hdr = "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:10.0.3) Gecko/20120305 Firefox/10.0.3\r\n";

int read_one_line(rio_t *rio, char *line) {
    int n = Rio_readlineb(rio, line, MAXLINE);
    assert(n >= 2);
    assert(line[n - 2] == '\r');
    assert(line[n - 1] == '\n');
    printf("< %s", line);
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
 * @brief Parse URI into filename and CGI args.
 * 
 * Borrowed from `tiny.c`.
 * 
 * @return 0 if dynamic content, 1 if static
 */
int parse_uri(char *uri, char *filename, char *cgiargs) 
{
    char *ptr;

    /* `strstr(haystack, needle)` finds the first occurrence of `needle` in `haystack` */
    if (!strstr(uri, "cgi-bin")) {  /* Static content */ //line:netp:parseuri:isstatic
        strcpy(cgiargs, "");                             //line:netp:parseuri:clearcgi
        strcpy(filename, ".");                           //line:netp:parseuri:beginconvert1
        strcat(filename, uri);                           //line:netp:parseuri:endconvert1
        if (uri[strlen(uri)-1] == '/')                   //line:netp:parseuri:slashcheck
            strcat(filename, "home.html");               //line:netp:parseuri:appenddefault
        return 1;
    }
    else {  /* Dynamic content */                        //line:netp:parseuri:isdynamic
        ptr = index(uri, '?');                           //line:netp:parseuri:beginextract
        if (ptr) {
            strcpy(cgiargs, ptr+1);
            *ptr = '\0';
        }
        else 
            strcpy(cgiargs, "");                         //line:netp:parseuri:endextract
        strcpy(filename, ".");                           //line:netp:parseuri:beginconvert2
        strcat(filename, uri);                           //line:netp:parseuri:endconvert2
        return 0;
    }
}

/**
 * @brief Parse URL into hostname, filename and CGI args
 * 
 * @return 0 if dynamic content, 1 if static
 */
int parse_url(char *url, char **uri, char *hostname, char *filename, char *cgiargs) 
{
    char *ptr;

    assert(strncmp(url, "http", 4) == 0);

    /* `strchr(str, ch)` finds the first occurrence of `ch` in `str` */
    ptr = strchr(url, '/');
    ptr += 2;
    *uri = strchr(ptr, '/');
    size_t len = *uri - ptr;
    strncpy(hostname, ptr, len);
    hostname[len] = '\0';

    return parse_uri(*uri, filename, cgiargs);
}

size_t min(size_t x, size_t y) {
  return x < y ? x : y;
}

int has_prefix(char const *line, char const *prefix) {
    return !strncmp(line, prefix, min(strlen(line), strlen(prefix)));
}

void forward_request(int client_fd, char const *method, char const *uri,
        char const *hostname, char const *buf) {
    char line[MAXLINE];
    rio_t rio; Rio_readinitb(&rio, client_fd);

    // Send the request line:
    sprintf(line, "%s %s HTTP/1.0\r\n", method, uri);
    printf("> %s", line);
    Rio_writen(client_fd, line, strlen(line));
    // Send the HOST header:
    sprintf(line, "Host: %s\r\n", hostname);
    printf("> %s", line);
    Rio_writen(client_fd, line, strlen(line));
    // Send the User-Agent header:
    printf("> %s", user_agent_hdr);
    Rio_writen(client_fd, (void *)user_agent_hdr, strlen(user_agent_hdr));
    // Send the Connection header:
    sprintf(line, "%s\r\n", "Connection: close");
    printf("> %s", line);
    Rio_writen(client_fd, line, strlen(line));
    // Send the Connection header:
    sprintf(line, "%s\r\n", "Proxy-Connection: close");
    printf("> %s", line);
    Rio_writen(client_fd, line, strlen(line));
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
            printf("~ %s", line);
        } else {
            printf("> %s", line);
            Rio_writen(client_fd, line, strlen(line));
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

void serve(int connfd) {
    // Read the entire HTTP request from the client and check whether the it is valid.
    size_t n = 0;
    char buf[MAXLINE], *line = buf;
    rio_t rio;

    Rio_readinitb(&rio, connfd);
    // Read the first line:
    char method[MAXLINE], url[MAXLINE], version[MAXLINE];
    n = read_one_line(&rio, line);
    sscanf(line, "%s %s %s", method, url, version);
    if (strcasecmp(method, "GET")) {
        clienterror(connfd, method, "501", "Not Implemented",
                    "Tiny Proxy does not implement this method");
        return;
    }
    printf("* method = \"%s\"\n", method);
    printf("* version = \"%s\"\n", version);
    printf("* url = \"%s\"\n", url);
    // Parse URL from GET request
    char *uri, hostname[MAXLINE], filename[MAXLINE], cgiargs[MAXLINE];
    int is_static = parse_url(url, &uri, hostname, filename, cgiargs);
    printf("  * uri = \"%s\"\n", uri);
    printf("    * hostname = \"%s\"\n", hostname);
    printf("    * filename = \"%s\"\n", filename);
    printf("    * cgiargs = \"%s\"\n", cgiargs);
    // Read other lines:
    do {
      line += n;  // n does not account '\0'
      n = read_one_line(&rio, line);
    } while (n != 2);
    assert(strcmp(line, "\r\n") == 0);
    assert(strlen(buf) <= MAXLINE);
    printf("Length of the request: %ld\n", strlen(buf));
    // Connect to the appropriate web server.
    int client_fd;
    char *port = strchr(hostname, ':');
    if (port) {  // port explicitly given, use it
      *port++ = '\0';
    }
    client_fd = Open_clientfd(hostname, port ? port : "80");
    if (port) {  // port explicitly given, use it
      *--port = ':';
    }
    printf("Connected to (%s)\n", hostname);
    // Request the object the client specified.
    forward_request(client_fd, method, uri, hostname, buf);
    Close(client_fd);
    // Read the server's response and forward it to the client.
    printf("Forward response from server (%s) to client\n", hostname);
    sprintf(buf, "HTTP/1.0 200 OK\r\n");
    printf("> %s", buf);
    Rio_writen(connfd, buf, strlen(buf));
    sprintf(buf, "Server: Tiny Proxy Server\r\n");
    printf("> %s", buf);
    Rio_writen(connfd, buf, strlen(buf));
    sprintf(buf, "Connection: close\r\n");
    printf("> %s", buf);
    Rio_writen(connfd, buf, strlen(buf));
    sprintf(buf, "\r\n");
    printf("> %s", buf);
    Rio_writen(connfd, buf, strlen(buf));
}

int main(int argc, char **argv)
{
    int listenfd, connfd;
    socklen_t clientlen;
    struct sockaddr_storage clientaddr;  /* Enough space for any address */
    char client_hostname[MAXLINE], client_port[MAXLINE];

    if (argc != 2) {
        fprintf(stderr, "usage: %s <port>\n", argv[0]);
        exit(0);
    }

    listenfd = Open_listenfd(argv[1]);
    while (1) {
        clientlen = sizeof(struct sockaddr_storage); 
        connfd = Accept(listenfd, (SA *)&clientaddr, &clientlen);
        Getnameinfo((SA *) &clientaddr, clientlen, client_hostname, MAXLINE, 
                    client_port, MAXLINE, 0);
        printf("Connected to (%s, %s)\n", client_hostname, client_port);
        serve(connfd);
        Close(connfd);
    }
    exit(0);
}
