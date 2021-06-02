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

![](https://csapp.cs.cmu.edu/3e/ics3/conc/conc4.pdf)

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

![](https://csapp.cs.cmu.edu/3e/ics3/conc/state.pdf)

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

|                    进程上下文切换                    |                       线程上下文切换                       |
| :--------------------------------------------------: | :--------------------------------------------------------: |
| ![](https://csapp.cs.cmu.edu/3e/ics3/ecf/switch.pdf) | ![](https://csapp.cs.cmu.edu/3e/ics3/conc/concthreads.pdf) |

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
- 【显式结束】调用 `pthread_exit()` 结束当前线程及进程。
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

## 3.7. 初始化线程<a href id="pthread_once"></a>

```c
#include <pthread.h>
pthread_once_t once_control = PTHREAD_ONCE_INIT;
int pthread_once(pthread_once_t *once_control,
                 void (*init_routine)(void));
```

- 首次调用 `pthread_once()` 会运行 `init_routine()` 以初始化全局变量。
- 用相同的 `once_control` 再次调用 `pthread_once()` 不会做任何事。

## 3.8. `echoservert.c`<a href id="echoserver-thread"></a>

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

【共享变量】被多个线程（直接或间接）访问的变量。

- 寄存器中的数据始终独享，虚拟内存中的数据可以共享。
- 各线程通常不访问其他线程的栈区，但栈区属于虚拟内存，故仍可共享。

```c
#include "csapp.h"
#define N 2
void *thread(void *vargp);

char **ptr;  /* 全局变量 in 数据读写区，直接共享 */

int main() {
  int i;  
  pthread_t tid;
  char *msgs[N] = { /* 局部自动变量 in 主线程栈区，间接共享 */
    "Hello from foo", "Hello from bar"
  };

  ptr = msgs;
  for (i = 0; i < N; i++)
    Pthread_create(&tid, NULL, thread, (void *)i);
  Pthread_exit(NULL);
}

void *thread(void *vargp) {
  int myid = (int)vargp; /* 局部自动变量 in 该线程栈区，不被共享 */
  static int count = 0;  /* 局部静态变量 in 数据读写区，直接共享 */
  printf("[%d]: %s (count=%d)\n", myid, ptr[myid], ++count);
  return NULL;
}
```

# 5. 用旗语同步线程

一般而言，无法预知各线程被操作系统选中的执行顺序。

![](https://csapp.cs.cmu.edu/3e/ics3/conc/badcntasm.pdf)

## 5.1. 进程图<a href id="graph"></a>

【进程图 (progress graph)】

- $n$ 个线程的执行过程对应于 $n$ 维空间中的轨迹。
- 第 $k$ 坐标轴对应于第 $k$ 线程。
- 点 $(I_1,I_2,\dots,I_n)$ 表示第 $k$ 线程完成指令 $I_k$ 后的状态，其中 $k=1,\dots,n$。
- （单核处理器）同一时间只能执行一条指令，故轨迹的生长始终平行于某一坐标轴。

进程图有助于理解以下概念：
- 【关键段 (critical section)】操纵共享变量的指令序列。<a href id="critical"></a>
- 【互斥 (mutual exclusion)】任一线程执行关键段时，应当暂时独享对共享变量访问。
- 【不安全区域 (unsafe region)】$n$ 维空间内的开集（不含边界），在第 $k$ 坐标轴上的投影为第 $k$ 线程的关键段。<a href id="unsafe"></a>
- 【不安全轨迹 (unsafe trajectory)】经过不安全区的轨迹，各线程对共享变量的访问会发生竞争。

![](https://csapp.cs.cmu.edu/3e/ics3/conc/safetraj.pdf)

## 5.2. 旗语<a href id="semaphore"></a>

【旗语 (semaphore)】用于同步并发程序的整型全局变量 `s`​，只能由以下方法修改（由🇳🇱计算机科学家 Dijkstra 发明）

- 【`P(s)​`】🇳🇱proberen🇨🇳检测
  - 若 `s != 0`，则 `return --s`，此过程不可被中断。
  - 若 `s == 0`，则暂停当前线程，直到被 `V(s)​` 重启，再 `return --s`。
- 【`V(s)​`】🇳🇱verhogen🇨🇳增加
  - 读取 `s`、`++s​`、存储 `s`，此过程不可被中断。
  - 若某些线程在 `P(s)​` 中等待，则重启其中任意一个。
- 【不变量】`s >= 0` 始终成立。

![](https://csapp.cs.cmu.edu/3e/ics3/conc/pgsem.pdf)

POSIX  标准定义了以下接口：

```c
#include <semaphore.h>
int sem_init(sem_t *sem, int pshared/* 通常为 0 */, unsigned int v/* 通常为 1 */);
int sem_wait(sem_t *s); /* P(s) */
int sem_post(sem_t *s); /* V(s) */
#include "csapp.h"
void P(sem_t *s); /* Wrapper function for sem_wait */
void V(sem_t *s); /* Wrapper function for sem_post */
```

## 5.3. 用旗语实现互斥

【二项旗语 (binary semaphore)】为每个共享变量关联一个初值为 `1` 的旗语 `s`，用 `P(s)` 及 `V(s)` 包围[关键段](#critical)。

- 【互斥 (mutex)】用于支持对共享变量“互斥 (MUTually EXclusive)”访问的二项旗语。
- 【上锁 (lock)】在关键段头部调用 `P(s)`
- 【开锁 (unlock)】在关键段尾部调用 `V(s)`
- 【禁止区域 (forbidden region)】`s < 0` 的区域，亦即[不安全区域](#unsafe)。

```c
#include "csapp.h"

volatile long count = 0; /* Counter */
sem_t mutex; /* Semaphore that protects `count` */

void *thread(void *vargp) {
  long n_iters = *((long *)vargp);
  for (long i = 0; i < n_iters; i++) {
    P(&mutex);
    count++;
    V(&mutex);
  }
  return NULL;
}

int main(int argc, char **argv) {
  long n_iters;
  pthread_t tid1, tid2;

  /* Check input argument */
  if (argc != 2) { 
    printf("usage: %s <niters>\n", argv[0]);
    exit(0);
  }
  n_iters = atoi(argv[1]);

  Sem_init(&mutex, 0, 1);
  /* Create threads and wait for them to finish */
  Pthread_create(&tid1, NULL, thread, &n_iters);
  Pthread_create(&tid2, NULL, thread, &n_iters);
  Pthread_join(tid1, NULL);
  Pthread_join(tid2, NULL);
  /* Check result */
  if (count != (2 * n_iters))
    printf("BOOM! count=%ld\n", count);
  else
    printf("OK count=%ld\n", count);
  exit(0);
}
```

## 5.4. 用旗语调度共享资源

【计数旗语 (counting semaphore)】

### 生产者--消费者

【有界缓冲区 (bounded buffer)】<a href id="bounded-buffer"></a>

- 【生产者 (producer)】
  - 若缓冲区有空，则向其中填入新“项目 (item)”；否则等待有空。
  - 实例：视频编码器、GUI 事件检测。
- 【消费者 (consumer)】
  - 若缓冲区非空，则从其中移出项目；否则等待非空。
  - 实例：视频解码器、GUI 事件响应。

```c
typedef struct {
  int *buf;          /* Buffer array */         
  int n;             /* Maximum number of slots */
  int front;         /* buf[(front+1)%n] is first item */
  int rear;          /* buf[rear%n] is last item */
  sem_t mutex;       /* Protects accesses to buf */
  sem_t slots;       /* Counts available slots */
  sem_t items;       /* Counts available items */
} sbuf_t;

/* Create an empty, bounded, shared FIFO buffer with n slots */
void sbuf_init(sbuf_t *sp, int n) {
  sp->buf = Calloc(n, sizeof(int)); 
  sp->n = n;                       /* Buffer holds max of n items */
  sp->front = sp->rear = 0;        /* Empty buffer iff front == rear */
  Sem_init(&sp->mutex, 0, 1);      /* Binary semaphore for locking */
  Sem_init(&sp->slots, 0, n);      /* Initially, buf has n empty slots */
  Sem_init(&sp->items, 0, 0);      /* Initially, buf has zero data items */
}

/* Clean up buffer sp */
void sbuf_deinit(sbuf_t *sp) {
  Free(sp->buf);
}

/* Insert item onto the rear of shared buffer sp */
void sbuf_insert(sbuf_t *sp, int item) {
  P(&sp->slots);                          /* Wait for available slot */
  P(&sp->mutex);                          /* Lock the buffer */
  sp->buf[(++sp->rear)%(sp->n)] = item;   /* Insert the item */
  V(&sp->mutex);                          /* Unlock the buffer */
  V(&sp->items);                          /* Announce available item */
}

/* Remove and return the first item from buffer sp */
int sbuf_remove(sbuf_t *sp) {
  int item;
  P(&sp->items);                          /* Wait for available item */
  P(&sp->mutex);                          /* Lock the buffer */
  item = sp->buf[(++sp->front)%(sp->n)];  /* Remove the item */
  V(&sp->mutex);                          /* Unlock the buffer */
  V(&sp->slots);                          /* Announce available slot */
  return item;
}
```

### 读者--作者

- 【读者 (reader)】
  - 只对共享资源进行读取的线程，可以与不限数量的读者共享资源。
  - 实例：网购时查看库存的用户、读取网页缓存的线程。
  - 第一类读写问题：偏向读者，读者无需等待作者写完。
- 【作者 (writer)】
  - 可对共享资源进行修改的线程，修改时只能独享对资源的访问权。
  - 实例：网购时正在下单的用户、更新网页缓存的线程。
  - 第二类读写问题：偏向作者，作者应尽快写出，读者需等待作者写完。

```c
/* 第一类读写问题解决方案 */
int readcnt;    /* 初值为 0 */
sem_t mutex, w; /* 初值为 1 */

void reader(void) {
  while (1) {
    P(&mutex);
    readcnt++;
    if (readcnt == 1)
    	P(&w); /* 第一个 reader 负责上锁 */
    V(&mutex);
    /* 关键段：多个线程可以并发读取 */
    P(&mutex);
    readcnt--;
    if (readcnt == 0)
    	V(&w); /* 最后一个 reader 负责开锁 */
    V(&mutex);
  }
}

void writer(void) {
  while (1) {
    P(&w);
    /* 关键段：至多一个 writer 在写 */
    V(&w);
  }
}
```

## 5.5. `echoservert-pre.c`

[`echoservert.c`](#echoserver-thread) 为每个客户端创建一个线程。为减少创建线程的开销，可采用以下方案：

- “主管线程 (master thread)”接收客户端发来的连接请求，再作为生产者向[有界缓冲区](#bounded-buffer)填入（套接字）描述符。
- “工人线程 (worker thread)”作为消费者从上述缓冲区移出（套接字）描述符，再响应客户端发来的文字信息。
- 工人线程数量 $\ll$ 缓冲区容量

![](https://csapp.cs.cmu.edu/3e/ics3/conc/prethreaded.pdf)

```c
#include "csapp.h"
#include "sbuf.h"
#define NTHREADS  4
#define SBUFSIZE  16

void echo_cnt(int connfd);
void *thread(void *vargp);

sbuf_t sbuf; /* Shared buffer of connected descriptors */

int main(int argc, char **argv) {
  int i, listen_fd, connect_fd;
  socklen_t client_len;
  struct sockaddr_storage client_addr;
  pthread_t tid; 

  if (argc != 2) {
    fprintf(stderr, "usage: %s <port>\n", argv[0]);
    exit(0);
  }
  listen_fd = Open_listenfd(argv[1]);

  sbuf_init(&sbuf, SBUFSIZE);
  for (i = 0; i < NTHREADS; i++)  /* Create worker threads */
    Pthread_create(&tid, NULL, thread, NULL);

  while (1) { 
    client_len = sizeof(struct sockaddr_storage);
    connect_fd = Accept(listen_fd, (SA *)&client_addr, &client_len);
    sbuf_insert(&sbuf, connect_fd); /* Insert connect_fd in buffer */
  }
}

void *thread(void *vargp) {
  Pthread_detach(pthread_self());
  while (1) {
    int connect_fd = sbuf_remove(&sbuf); /* Remove connect_fd from buffer */
    echo_cnt(connect_fd);                /* Service client */
    Close(connect_fd);
  }
}
```

### `echo_cnt.c`

`echo_cnt()`  用到了 [`pthread_once()`](#pthread_once)：

```c
#include "csapp.h"

static int byte_cnt;  /* Byte counter ... */
static sem_t mutex;   /* and the mutex that protects it */

static void init_echo_cnt(void) {
  Sem_init(&mutex, 0, 1);
  byte_cnt = 0;
}

void echo_cnt(int connect_fd) {
  int n; 
  char buf[MAXLINE]; 
  rio_t rio;
  static pthread_once_t once = PTHREAD_ONCE_INIT;

  Pthread_once(&once, init_echo_cnt); /* 初始化，只运行一次 */
  Rio_readinitb(&rio, connect_fd);
  while((n = Rio_readlineb(&rio, buf, MAXLINE)) != 0) {
    P(&mutex);
    byte_cnt += n;
    printf("server received %d (%d total) bytes on fd %d\n", 
           n, byte_cnt, connect_fd);
    V(&mutex);
    Rio_writen(connect_fd, buf, n);
  }
}
```

# 6. 多线程并行<a href id="parallel"></a>

【并行程序 (parallel program)】运行在“多核处理器 (multi-core processor)”上的“并发程序 (concurrent program)”。

【通用技巧】将原问题划分为若干子问题，各线程依据其 TID 计算相应的子问题。

$$
\sum_{i=0}^{mn-1}f(i)=\sum_{k=0}^{m-1}\left(\sum_{i=kn+0}^{kn+n-1}f(i)\right)
$$

## `psum-mutex.c`

```c
#include "csapp.h"
#define MAXTHREADS 32

long gsum = 0;           /* Global sum */
long nelems_per_thread;  /* Number of elements to sum */
sem_t mutex;             /* Mutex to protect global sum */

void *sum_mutex(void *vargp) {
  long myid = *((long *)vargp);          /* Extract the thread ID */
  long start = myid * nelems_per_thread; /* Start element index */
  long end = start + nelems_per_thread;  /* End element index */

  for (long i = start; i < end; i++) {
    P(&mutex); gsum += i; V(&mutex); /* 同步次数太多 ⚠️ */
  }
  return NULL;
}

int main(int argc, char **argv) {
  long i, nelems, log_nelems, nthreads, myid[MAXTHREADS];
  pthread_t tid[MAXTHREADS];
  nelems_per_thread = /* ... */
    sem_init(&mutex, 0, 1);

  /* Create peer threads and wait for them to finish */
  for (i = 0; i < nthreads; i++) {
    myid[i] = i;
    Pthread_create(&tid[i], NULL, sum_mutex, &myid[i]);
  }
  for (i = 0; i < nthreads; i++)
    Pthread_join(tid[i], NULL);
  
  /* ... */
}
```

【重要结论】同步开销很昂贵，应当尽量避免；若不能避免，则单次同步应当“分摊 (amortize)”尽可能多的计算量。

## `psum-array.c`

```c
#include "csapp.h"
#define MAXTHREADS 32

long psum[MAXTHREADS];  /* Partial sum computed by each thread */
long nelems_per_thread; /* Number of elements summed by each thread */

void *sum_array(void *vargp) {
  long myid = *((long *)vargp);          /* Extract the thread ID */
  long start = myid * nelems_per_thread; /* Start element index */
  long end = start + nelems_per_thread;  /* End element index */

  for (long i = start; i < end; i++)
    psum[myid] += i; /* 不必同步，但访存太多 ⚠️ */
  return NULL;
}

int main(int argc, char **argv) {
  long i, nelems, log_nelems, nthreads, myid[MAXTHREADS];
  pthread_t tid[MAXTHREADS];

  nelems_per_thread = /* ... */

  /* Create peer threads and wait for them to finish */
  for (i = 0; i < nthreads; i++) {
    myid[i] = i;
    Pthread_create(&tid[i], NULL, sum_array, &myid[i]);
  }
  for (i = 0; i < nthreads; i++)
    Pthread_join(tid[i], NULL);

  /* Add up the partial sums computed by each thread */
  for (i = 0; i < nthreads; i++)
    result += psum[i];

  /* ... */
}

```

## `psum-local.c`

```c
/* 同 psum-array.c */

void *sum_array(void *vargp) {
  /* 同 psum-array.c */

  long sum = 0; /* 寄存器变量 */
  for (long i = start; i < end; i++)
    sum += i; /* 不必同步，且不必访存 */
  psum[myid] = sum; /* 只访问一次 */
  return NULL;
}

/* 同 psum-array.c */
```

【实验现象】核心数量 `n_cores` 等于线程数量 `n_threads` 时，加速效果最好。

## 并行程序性能

性能指标

- 【加速 (speedup)】$S_p=T_1/T_p$
  - 【绝对加速 (absolute speedup)】若 $T_1$ 为串行版本（在单核上）的运行时间
  - 【相对加速 (relative speedup)】若 $T_1$ 为并行版本在单核上的运行时间
- 【效率 (efficiency)】$E_p=S_p/p\equiv T_1/(pT_p)$

可扩展性

- 【强可扩展 (strong scalable)】问题规模不变，加速正比于核心数量。
  - 例：处理固定数量的传感器传回的信号。
- 【弱可扩展 (weak scalable)】耗时基本不变，问题规模正比于核心数量。
  - 例：科学计算程序。

# 7. 其他并发问题

## 7.1. 线程安全<a href id="thread-safe"></a>

【线程安全 (thread-safe)】反复运行并发的多线程函数总是给出相同结果。

以下函数不是线程安全的：

1. 没有保护共享变量
   - 解决：用[旗语](#semaphore)加锁
   - 优点：调用侧无需改动
   - 缺点：性能下降
2. 有依赖于调用历史的状态（全局或静态变量）
   - 实例：伪随机数
   - 解决：改写为[再入函数](#reentrant)
3. 返回指向静态变量的指针
   - 实例：`ctime()`, `gethostname()`
   - 解决：由调用侧传入指向私有变量的指针；【锁后复制 (lock-and-copy)】将不安全函数封装在 `P/V` 对内，在其中将结果（深）复制到私有内存。
4. 调用了第 2 类不安全函数
   - 解决：锁后复制

## 7.2. 再入函数<a href id="reentrant"></a>

【再入函数 (reentrant function)】被多个线程调用时，不访问共享变量。

- 一定是线程安全的
- 比加锁的函数更高效
- 若传入指针，则需指向调用侧的私有数据

```c
/* rand_r - return a pseudorandom integer on 0..32767 */
int rand_r(unsigned int *nextp/* 指向调用侧的私有数据 */) {
  *nextp = *nextp * 1103515245 + 12345;
  return (unsigned int)(*nextp / 65536) % 32768;
}
```

## 7.3. 标准库函数

大多数 C 标准库函数是线程安全的。部分不安全的有[再入](#reentrant)版本（以 `_r` 为后缀）：

- 第 2 类（必须用改写版本）
  - `rand()`
  - `strtok()`
- 第 3 类（可用锁后复制，但低效）
  - `asctime()`, `ctime()`, `localtime()`
  - `gethostbyaddr()`, `gethostbyname()`
  - `inet_ntoa()` ⚠️ 无再入版本

## 7.4. 竞争

【竞争 (race)】结果依赖于[进程图](#graph)中的路径。

## 7.5. 死锁

![](https://csapp.cs.cmu.edu/3e/ics3/conc/deadlock.pdf)

【死锁 (deadlock)】某些被暂停的进程等待着不可能发生的事件。

- 解决：确保各线程对各锁的上锁顺序一致。

