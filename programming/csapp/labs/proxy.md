---
title: Proxy Lab
---

# Download

- [`proxylab.pdf`](./proxy/proxylab.pdf) specifies the requirements.
- [`proxylab-handout.tar`](https://csapp.cs.cmu.edu/3e/proxylab-handout.tar) provides the auto-grader and helper code (e.g. `csapp.h`, `csapp.c`, and the `tiny` folder).

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

The test on this part
- opens a blocking nop-server that never responds.
- requests a file from the nop-server, which never returns.
- requests a file from the Tiny server, which should return immediately.
- requests a file from the proxy, which should return the same thing the Tiny server returns.

## 2.1 Create-on-Request

So, a simple create-on-request policy is enough for passing the test on this part:

```c
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
```

## 2.2 Precreated threads (TODO)

A more efficient but complicated version like [`echoservert-pre.c`](../12_concurrent_programming.md#echoservert-pre) in the textbook is also possible.

# 3. Caching web objects

## 3.1 LRU based on [`uthash`](http://troydhanson.github.io/uthash/userguide.html) and [`utlist`](troydhanson.github.io/uthash/utlist.html)

Types:

```c
struct _node;
typedef struct _node node_t;

struct _item;
typedef struct _item item_t;

struct _lru;
typedef struct _lru lru_t;

struct _item {
    char *key;  // URI from a client
    struct {
      node_t *node;  // node in a `list`
      char const *data;  // response from a server
      int size;  // size of the response
    } value;
    UT_hash_handle hh;  /* makes this structure hashable */
};

struct _node {
    item_t *item;
    node_t *prev, *next; /* needed for doubly-linked lists */
};

struct _lru {
    item_t *map;
    node_t *list;
    int capacity;  // max size of the sum of all item->value.data
    int size;  // current size of the sum of all item->value.data
};
```

Methods:

```c
lru_t *lru_construct(int capacity);
void lru_destruct(lru_t *lru);

item_t *lru_find(lru_t const *lru, char const *key);
void lru_print(lru_t const *lru);

void lru_emplace(lru_t *lru, char const *key, char const *data, int size);
void lru_sink(lru_t *lru, item_t *item);
void lru_pop(lru_t *lru);
```

## 3.2 Using [`pthread_rwlock_t`](../12_concurrent_programming.md#pthread_rwlock_t)

`lru_find()` is read-only:

```c
pthread_rwlock_rdlock(&lru_rwlock);
item_t *item = lru_find(lru, uri_from_client);
if (item) {
  /* .. */
  Rio_writen(client_fd, item->value.data, item->value.size);
}
pthread_rwlock_unlock(&lru_rwlock);
```

`lru_emplace()` is a writing operation:

```c
pthread_rwlock_wrlock(&lru_rwlock);
// The item might already been emplaced by another thread, so find it again:
if (!lru_find(lru, uri_from_client)) {
  lru_emplace(lru, uri_from_client, buf, size);
}
pthread_rwlock_unlock(&lru_rwlock);
```

`lru_sink()` is also a writing operation:

```c
pthread_rwlock_wrlock(&lru_rwlock);
// The item might already been popped by another thread, so find it again:
if ((item = lru_find(lru, uri_from_client)) != NULL) {
    lru_sink(lru, item);
}
pthread_rwlock_unlock(&lru_rwlock);
```
