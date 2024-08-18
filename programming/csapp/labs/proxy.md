---
title: Proxy Lab
---

# Download

- [`proxylab.pdf`](./proxy/proxylab.pdf) specifies the requirements.
- [`proxylab-handout.tar`](https://csapp.cs.cmu.edu/3e/proxylab-handout.tar) provides the auto-grader and helper code.

# Build and Run

```shell
# Get my port number (got 8982):
./port-for-user.pl pvc
# Build and start the tiny (web) server:
cd tiny
make && ./tiny 8981 &
cd -
# Build and start the proxy server:
make && ./proxy 8982 &
# Use the proxy via curl:
curl -v --proxy http://localhost:8982 http://localhost:8981/home.html
curl -v --proxy http://localhost:8982 http://localhost:8981/cgi-bin/adder\?15000\&213
curl -v --proxy http://localhost:8982 http://localhost:8981/cgi-bin/minus\?15000\&213  # triggers 404
# Auto grade the solution:
./driver.sh
```

# 1. Implementing a sequential web proxy

## 1.0 Initialize the Server

Just mock [`echoserveri.c`](../code/netp/echoserveri.c) and [`echo.c`](../code/netp/echo.c).

## 1.1 Parse the URI

```c
char const */* the address of the first char in uri_to_server */
parse_uri(char const *uri_from_client, char *hostname/* output */);
```

Note that `uri_to_server` is the suffix of `uri_from_client`.

### ⚠️ Avoid modifying URI in `parse_uri()`

The `parse_uri()` implemented in [`tiny.c`](./proxy/tiny/tiny.c) modifies URI:

```c
char *ptr = index(uri, '?');
*ptr = '\0';
strcat(filename, uri);
```

It can be avoided by using `strncat()` instead of `strcat()`:

```c
char const *ptr = index(uri, '?');
strncat(filename, uri, ptr - uri);
```

## 1.2 Connect to the server

```c
int/* server_fd */ connect_to_server(char *hostname);
```

## 1.3 Forward request from the client to the server

Replace the headers begin with `GET` or `Host` or `User-Agent` or `Connection` or `Proxy-Connection`.

Send other headers unmodified.

```c
void forward_request(int server_fd,
    char const *method/* GET */,
    char const *uri_to_server,
    char const *hostname,
    char const *buf/* the buffer holding request_from_client */);
```

## 1.4 Forward response from the server to the client

```c
void forward_response(int server_fd, int client_fd,
    char const *uri_from_client/* the key for cache */,
    char *buf/* buffer for holding reponse_from_server */);
```

The argument `uri_from_client` is not necessary here, but is useful for implementing the LRU cache.

# 2. Dealing with multiple concurrent requests

# 3. Caching web objects

