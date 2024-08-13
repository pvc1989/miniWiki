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

void serve(int connfd) {
    // Read the entire HTTP request from the client and check whether the it is valid.
    size_t n = 0;
    char buf[MAXLINE], *line = buf;
    rio_t rio;

    Rio_readinitb(&rio, connfd);
    // Read the first line:
    char method[MAXLINE], uri[MAXLINE], version[MAXLINE];
    n = read_one_line(&rio, line);
    sscanf(line, "%s %s %s", method, uri, version);
    if (strcasecmp(method, "GET")) {
        clienterror(connfd, method, "501", "Not Implemented",
                    "Tiny Proxy does not implement this method");
        return;
    }
    // Read other lines:
    do {
      line += n;  // n does not account '\0'
      n = read_one_line(&rio, line);
    } while (n != 2);
    assert(strcmp(line, "\r\n") == 0);
    assert(strlen(buf) <= MAXLINE);
    printf("Length of the request: %ld\n", strlen(buf));
    // Connect to the appropriate web server.

    // Request the object the client specified.

    // Read the server's response and forward it to the client.
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
