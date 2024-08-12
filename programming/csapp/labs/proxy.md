---
title: Proxy Lab
---

# Build and Run

```shell
# Get my port number:
./port-for-user.pl pvc
# Build the project:
make
# Start the proxy server:
./proxy 8982
# Auto grade the solution:
./driver.sh
```

# 1. Implementing a sequential web proxy

## 1.0 Initialize the Server

Just mock [`echoserveri.c`](../code/netp/echoserveri.c) and [`echo.c`](../code/netp/echo.c).

# 2. Dealing with multiple concurrent requests

# 3. Caching web objects

