---
title: 并发编程
---

【并发程序 (concurrent program)】采取应用级并发的应用程序。

# 1. 基于进程的并发

1. 服务器收到一个客户端 `Client_1` 发来的连接请求。
   - 返回一个异于 `listen_fd` 的 `connect_fd_1`
2. 服务器 `fork` 出一个子进程 `Child_1`，由后者向 `Client_1` 提供服务。
   - 子进程 `Child_1` 关闭 `listen_fd`
   - 主进程 `Parent` 关闭 `connect_fd_1`
3. 服务器收到另一个客户端 `Client_2` 发来的连接请求。
   - 返回一个异于 `listen_fd` 的 `connect_fd_2`
4. 服务器 `fork` 出另一个子进程 `Child_2`，由后者向 `Client_2` 提供服务。
   - 子进程 `Child_2` 关闭 `listen_fd`
   - 主进程 `Parent` 关闭 `connect_fd_2`

## 1.1. `echoserverp.c`

```c
#include "csapp.h"
void echo(int connect_fd);

void sigchld_handler(int sig) {
  while (waitpid(-1, 0, WNOHANG) > 0)
    ;
  return;
}

int main(int argc, char **argv) {
  int listen_fd, connect_fd;
  socklen_t client_len;
  struct sockaddr_storage client_addr;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <port>\n", argv[0]);
    exit(0);
  }

  Signal(SIGCHLD, sigchld_handler);
  listen_fd = Open_listenfd(argv[1]);
  while (1) {
    client_len = sizeof(struct sockaddr_storage); 
    connect_fd = Accept(listen_fd, (SA *)&client_addr, &client_len);
    if (Fork() == 0) { 
      Close(listen_fd);   /* Child closes its listening socket */
      echo(connect_fd);   /* Child services client */
      Close(connect_fd);  /* Child closes connection with client */
      exit(0);            /* Child exits */
    }
    Close(connect_fd); /* Parent closes connected socket (important!) */
  }
}
```

## 1.2. 进程的优缺点

各进程有独立的虚拟内存空间，既是优点，也是缺点：

- 【优点】各进程只能读写自己的虚拟内存，不会破坏其他进程的虚拟内存。
- 【缺点】进程之间共享数据变得困难，必须使用显式“进程间通信 (InterProcess Communication, IPC)”。

# 2. 基于读写复用的并发

## `select()`

【需求】并发处理“连接请求”与“键盘输入”：

- 等待连接请求，会屏蔽键盘输入。
- 等待键盘输入，会屏蔽连接请求。

【方案】用 `select()` 实现“读写复用 (I/O multiplexing)”。

```c
#include <sys/select.h>
int select(int n/* 集合大小 */, fd_set *fdset, NULL, NULL, NULL);
FD_ZERO(fd_set *fdset);          /* Clear all bits in `fdset` */
FD_CLR(int fd, fd_set *fdset);   /* Clear bit `fd` in `fdset` */
FD_SET(int fd, fd_set *fdset);   /* Turn on bit `fd` in `fdset` */
FD_ISSET(int fd, fd_set *fdset); /* Is bit `fd` in `fdset` on? */
```

此函数令内核暂停当前进程，直到“读取集 (read set)” `fdset` 中的至少一个（文件或套接字）描述符可以被读取，返回“可用集 (ready set)”的“基数 (cardinality)”，并将“读取集”修改为“可用集”。

```c
#include "csapp.h"
void echo(int connect_fd);
void command(void);

int main(int argc, char **argv) {
  int listen_fd, connect_fd;
  socklen_t client_len;
  struct sockaddr_storage client_addr;
  fd_set read_set, ready_set;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <port>\n", argv[0]);
    exit(0);
  }
  listen_fd = Open_listenfd(argv[1]);

  FD_ZERO(&read_set);              /* read_set = { } */
  FD_SET(STDIN_FILENO, &read_set); /* read_set = { `stdin` } */
  FD_SET(listen_fd, &read_set);    /* read_set = { `stdin`, `listen_fd` } */

  while (1) {
    ready_set = read_set;
    Select(listen_fd+1, &ready_set, NULL, NULL, NULL);
    /* 直到 `stdin` 或 `listen_fd` 可用 */
    if (FD_ISSET(STDIN_FILENO, &ready_set)) {
      /* `stdin` 可用，响应键盘输入 */
      command();
    }
    if (FD_ISSET(listen_fd, &ready_set)) {
      /* `listen_fd` 可用，响应连接请求 */
      client_len = sizeof(struct sockaddr_storage); 
      connect_fd = Accept(listen_fd, (SA *)&client_addr, &client_len);
      echo(connect_fd); /* 可优化为 `echo_at_most_one_line()` */
      Close(connect_fd);
    }
  }
}

void command(void) {
  char buf[MAXLINE];
  if (!Fgets(buf, MAXLINE, stdin))
    exit(0); /* EOF */
  printf("%s", buf); /* Process the input command */
}
```

## 2.1. `echoservers.c`

【状态机 (state machine)】服务器为客户端 `Client_k` 分配描述符 `fd_k`

- 【状态 (state)】服务器等待描述符 `fd_k` 可用。
- 【事件 (event)】描述符 `fd_k` 可用，服务器通过 `select()` 检测。
- 【迁移 (transition)】服务器从 `fd_k` 读取一行，通过 `check_clients()` 实现。

```c
#include "csapp.h"

typedef struct { /* Represents a pool of connected descriptors */
  int max_fd;        /* Largest descriptor in read_set */   
  fd_set read_set;   /* Set of all active descriptors */
  fd_set ready_set;  /* Subset of descriptors ready for reading  */
  int n_ready;       /* Number of ready descriptors from select */   
  int max_i;         /* Highwater index into client array */
  int client_fd[FD_SETSIZE];    /* Set of active descriptors */
  rio_t client_rio[FD_SETSIZE]; /* Set of active read buffers */
} pool_t;

void init_pool(int listen_fd, pool_t *p);
void add_client(int connect_fd, pool_t *p);
void check_clients(pool_t *p);

int byte_count = 0; /* Counts total bytes received by server */

int main(int argc, char **argv) {
  int listen_fd, connect_fd;
  socklen_t client_len;
  struct sockaddr_storage client_addr;
  static pool_t pool; 

  if (argc != 2) {
    fprintf(stderr, "usage: %s <port>\n", argv[0]);
    exit(0);
  }
  listen_fd = Open_listenfd(argv[1]);
  init_pool(listen_fd, &pool);

  while (1) {
    /* Wait for listening/connected descriptor(s) to become ready */
    pool.ready_set = pool.read_set;
    pool.n_ready = Select(pool.max_fd+1, &pool.ready_set, NULL, NULL, NULL);

    /* If listening descriptor ready, add new client to pool */
    if (FD_ISSET(listen_fd, &pool.ready_set)) {
      client_len = sizeof(struct sockaddr_storage);
      connect_fd = Accept(listen_fd, (SA *)&client_addr, &client_len);
      add_client(connect_fd, &pool);
    }

    /* Echo a text line from each ready connected descriptor */ 
    check_clients(&pool);
  }
}

void init_pool(int listen_fd, pool_t *p) {
  /* Initially, there are no connected descriptors */
  int i;
  p->max_i = -1;
  for (i = 0; i < FD_SETSIZE; i++)  
    p->client_fd[i] = -1;

  /* Initially, listen_fd is the only member of read_set */
  p->max_fd = listen_fd;
  FD_ZERO(&p->read_set);
  FD_SET(listen_fd, &p->read_set);
}

void add_client(int connect_fd, pool_t *p) {
  int i;
  p->n_ready--;
  for (i = 0; i < FD_SETSIZE; i++) {
    if (p->client_fd[i] < 0) { /* Find an available slot */
      /* Add connect_fd to the pool */
      p->client_fd[i] = connect_fd;
      Rio_readinitb(&p->client_rio[i], connect_fd);

      FD_SET(connect_fd, &p->read_set); /* Add connect_fd to read_set */

      /* Update max_fd and max_i */
      if (connect_fd > p->max_fd)
        p->max_fd = connect_fd;
      if (i > p->max_i)
        p->max_i = i;
      break;
    }
  }
  if (i == FD_SETSIZE) /* Couldn't find an empty slot */
    app_error("add_client error: Too many clients");
}

void check_clients(pool_t *p) {
  int i, connect_fd, n;
  char buf[MAXLINE];
  rio_t rio;

  for (i = 0; (i <= p->max_i) && (p->n_ready > 0); i++) {
    connect_fd = p->client_fd[i];
    rio = p->client_rio[i];

    /* If the descriptor is ready, echo a text line from it */
    if ((connect_fd > 0) && (FD_ISSET(connect_fd, &p->ready_set))) { 
      p->n_ready--;
      if ((n = Rio_readlineb(&rio, buf, MAXLINE)) != 0) {
        byte_count += n;
        printf("Server received %d (%d total) bytes on fd %d\n", 
               n, byte_count, connect_fd);
        Rio_writen(connect_fd, buf, n);
      }
      else { /* EOF detected, remove descriptor from pool */
        Close(connect_fd);
        FD_CLR(connect_fd, &p->read_set);
        p->client_fd[i] = -1;
      }
    }
  }
}
```

## 2.2. 读写复用的优缺点

- 【优点】容易对不同客户端提供差异化服务；容易共享数据；比进程上下文切换更高效。
- 【缺点】粒度越细代码越复杂；易受不完整输入攻击；难以充分利用多核处理器。

# 3. 基于线程的并发

【线程 (thread)】运行在某个进程上下文中的一条逻辑控制流。

- 各线程有其独享的“线程上下文 (thread context)”（“线程号 (Thread ID, TID)”、运行期栈、通用寄存器、条件码）。
- 各线程共享其所属的“进程上下文 (process context)”（代码、数据、堆、共享库、打开的文件）。

## 3.1. 线程执行模型

线程执行模型与进程执行模型类似，但有以下区别：

- 线程上下文比进程上下文小很多，因此切换起来更快。
- 同一进程的各线程之间没有严格的主从关系。
  - 【主线程 (main thread)】最先运行的那个线程。
  - 【同伴进程 (peer thread)】除主线程外的其他线程。
  - 【同伴池 (a pool of peers)】同一进程的所有线程。

## 3.2. `pthread`

详见 [`man pthread`](https://man7.org/linux/man-pages/man7/pthreads.7.html)。

```c
#include "csapp.h"
void *thread(void *vargp) { /* thread routine */
  printf("Hello, world!\n");
  return NULL;
}
int main() {
  pthread_t tid;
  Pthread_create(&tid, NULL, thread, NULL); /* 创建同伴线程，在其中运行 `thread()` */
  Pthread_join(tid, NULL); /* 等待同伴线程结束 */
  exit(0);
}
```

`thread()` 只能接收与返回 `void*`，若要传入或返回多个参数，需借助 `struct`。

## 3.3. 创建线程

```c
#include <pthread.h>
typedef void *(func)(void *);
int pthread_create(pthread_t *tid, pthread_attr_t *attr/* NULL 表示默认属性 */,
                   func *f, void *arg/* 传给 f() 的实参 */);
pthread_t pthread_self(void); /* 返回当前线程的 TID */
```

## 3.4. 结束线程

结束线程的几种方式：

- 【隐式结束】传给 `pthread_create()` 的 `f()` 运行完毕并返回。
- 【显式结束】调用 `pthread_exit()` 结束当前线程，主线程会等待同伴线程结束。
- 【结束进程】某个同伴线程调用 `exit()` 结束整个进程。
- 【取消线程】因另一个进程调用 `pthread_cancel()` 而结束。

```c
#include <pthread.h>
void pthread_exit(void *thread_return);
int pthread_cancel(pthread_t tid);
```

## 3.5. 收割线程

```c
#include <pthread.h>
int pthread_join(pthread_t tid, void **thread_return);
```

与[收割子进程](./8_exceptional_control_flow.md#收割子进程)的 `waitpid()` 类似，但 `pthread_join()` 只能收割特定的线程。

## 3.6. 分离线程

任何线程总是处于以下两种状态之一：

- 【可加入的 (joinable)】可以被其他线程收割或取消，其内存资源在该线程被收割或取消时才被释放。（默认）
- 【分离的 (detached)】不能被其他线程收割或取消，其内存资源在该线程结束时被系统自动释放。（推荐）

为避免内存泄漏，任何可加入线程都应当被显式收割或取消，或通过以下函数转为分离的状态：

```c
#include <pthread.h>
int pthread_detach(pthread_t tid);
/* 常用：分离当前线程 */
pthread_detach(pthread_self());
```

## 3.7. 初始化线程

```c
#include <pthread.h>
pthread_once_t once_control = PTHREAD_ONCE_INIT;
int pthread_once(pthread_once_t *once_control,
                 void (*init_routine)(void));
```

- 首次调用 `pthread_once()` 会运行 `init_routine()` 以初始化全局变量。
- 用相同的 `once_control` 再次调用 `pthread_once()` 不会做任何事。

## 3.8. `echoservert.c`

```c
#include "csapp.h"

void echo(int connect_fd);

void *thread(void *vargp) { /* Thread routine */
  int connect_fd = *((int *)vargp);
  Pthread_detach(pthread_self());
  Free(vargp); /* malloc'ed in main thread */
  echo(connect_fd);
  Close(connect_fd);
  return NULL;
}

int main(int argc, char **argv) {
  int listen_fd, *connect_fdp;
  socklen_t client_len;
  struct sockaddr_storage client_addr;
  pthread_t tid; 

  if (argc != 2) {
    fprintf(stderr, "usage: %s <port>\n", argv[0]);
    exit(0);
  }
  listen_fd = Open_listenfd(argv[1]);

  while (1) {
    client_len = sizeof(struct sockaddr_storage);
    connect_fdp = Malloc(sizeof(int)); /* 若存于主线程的栈，会造成两个同伴线程的竞争 */
    *connect_fdp = Accept(listen_fd, (SA *)&client_addr, &client_len);
    Pthread_create(&tid, NULL, thread, connect_fdp/* 指向 connect_fd */);
  }
}
```

# 4. 多线程共享变量

# 5. 多线程同步

# 6. 多线程并行

# 7. 其他并发问题



