---
title: Proxy Lab
---

# Build and Run

```shell
# Get my port number (got 8982):
./port-for-user.pl pvc
# Build the project:
make
# Start the tiny (web) server:
cd tiny
./tiny 8981 &
cd -
# Start the proxy server:
./proxy 8982 &
# Use the proxy via curl:
curl -v --proxy http://localhost:8982 http://localhost:8981/home.html
# Auto grade the solution:
./driver.sh
```

# 1. Implementing a sequential web proxy

## 1.0 Initialize the Server

Just mock [`echoserveri.c`](../code/netp/echoserveri.c) and [`echo.c`](../code/netp/echo.c).

# ⚠️ Avoid modifying URI in `parse_uri()`

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

# 2. Dealing with multiple concurrent requests

# 3. Caching web objects

