---
title: Shell Lab
---

# Build & Run

```shell
make && ./sdriver.pl -t trace15.txt -s ./tshref -a "-p -v"
make && ./sdriver.pl -t trace15.txt -s ./tsh    -a "-p -v"
```

# `trace*.txt`

## 1. Properly terminate on EOF.

Do nothing.

## 2. Process builtin `quit` command.

```c
int builtin_cmd(char **argv) {
    if (!strcmp(argv[0], "quit"))
        exit(0);
    /* other builtin commands */
    return 0;
}
```

## 3. Run a FG job.

⚠️ `addjob()` and `deletejob()` has more arguments than the textbook version.

```c
void eval(char *cmdline) 
{
    char *argv[MAXARGS];
    int bg;
    pid_t pid;
    sigset_t mask_all, mask_one, prev_one;

    bg = parseline(cmdline, argv);
    if (argv[0] == NULL)
        return; /* ignore empty cmdline */

    if (builtin_cmd(argv)) 
        return;

    Sigfillset(&mask_all);
    /* Block SIGCHLD */
    Sigemptyset(&mask_one);
    Sigaddset(&mask_one, SIGCHLD);
    Sigprocmask(SIG_BLOCK, &mask_one, &prev_one);
    if ((pid = Fork()) == 0) { /* Child Process */
        Sigprocmask(SIG_SETMASK, &prev_one, NULL); /* Unblock SIGCHLD */
        Setpgid(0, 0); /* Put the child in a new process group whose ID
                           == the child's PID. */
        if (execve(argv[0], argv, environ) < 0) {
            printf("%s: Command not found\n", argv[0]);
            exit(1);
        }
    }
    Sigprocmask(SIG_BLOCK, &mask_all, NULL); /* Block all */
    addjob(jobs, pid, 1+bg, cmdline);
    Sigprocmask(SIG_SETMASK, &prev_one, NULL); /* recover */
    if (!bg)
        waitfg(pid);
}
```

In `waitfg()`, use `sigsuspend()` to wait:

```c
void waitfg(pid_t pid)
{
    sigset_t mask_all, prev_all;
    pid_t fg_pid;

    Sigfillset(&mask_all);

    while (1) {
        Sigprocmask(SIG_BLOCK, &mask_all, &prev_all);
        fg_pid = fgpid(jobs);
        if (fg_pid == pid) {
            Sigsuspend(&prev_all);
            Sigprocmask(SIG_SETMASK, &prev_all, NULL);
        }
        else {
            Sigprocmask(SIG_SETMASK, &prev_all, NULL);
            break;
        }
    }

    if (verbose) {
        printf("waitfg: Process (%d) no longer the fg process\n", pid);
        fflush(stdout);
    }
}
```

## 4. Run a BG job.

```c
void eval(char *cmdline) {
    /* ... */
    Sigprocmask(SIG_BLOCK, &mask_all, NULL); /* Block all */
    addjob(jobs, pid, 1+bg, cmdline);
    if (bg) {
        printf("[%d] (%d) %s", pid2jid(pid), pid, cmdline);
        fflush(stdout);
    }
    Sigprocmask(SIG_SETMASK, &prev_one, NULL); /* recover */
    if (!bg)
        waitfg(pid);
}
```

## 5. Process `jobs` builtin command.

```c
int builtin_cmd(char **argv) {
    sigset_t mask_all, prev_all;
    Sigfillset(&mask_all);

    /* ... */
    if (!strcmp(argv[0], "jobs")) {
        Sigprocmask(SIG_BLOCK, &mask_all, &prev_all);
        listjobs(jobs);
        Sigprocmask(SIG_SETMASK, &prev_all, NULL);
        return 1;
    }
    /* ... */
}
```

## 6. Forward `SIGINT` to FG job.

```c
void sigint_handler(int sig) {
    int olderrno = errno;
    sigset_t mask_all, prev_all;
    pid_t pid;
    char buf[MAXLINE]; int pos = 0; /* local buffer for sprintf() */

    if (verbose)
        Sio_puts("sigint_handler: entering\n");

    Sigfillset(&mask_all);
    Sigprocmask(SIG_BLOCK, &mask_all, &prev_all);
    pid = fgpid(jobs);
    Sigprocmask(SIG_SETMASK, &prev_all, NULL);
    Kill(pid, sig); /* to be improved in #11 */

    if (verbose) {
        pos += sprintf(buf + pos, "sigint_handler: Job (%d) killed\n", pid);
        pos += sprintf(buf + pos, "sigint_handler: exiting\n");
        Sio_puts(buf);
    }

    errno = olderrno;
}
```

## 7. Forward `SIGINT` only to FG job.

Do nothing.

## 8. Forward `SIGTSTP` only to FG job.

Similar to #6, just one more thing:

```c
void sigtstp_handler(int sig) {
    /* ... */
    job->state = ST; /* to be improved in #16 */
    /* ... */
}
```

## 9. Process `bg` builtin command

## 10. Process `fg` builtin command

## 11. Forward `SIGINT` to every process in FG process group

Similar to #6, just one change:

```c
Kill(-pid, sig); /* forward signal to FG process group (see #6) */
```

## 12. Forward `SIGTSTP` to every process in FG process group

## 13. Restart every stopped process in process group

## 14. Simple error handling

## 15. Putting it all together

Main test.

## 16. Process signals from other processes

Move `job->state = ST;` from `sigtstp_handler()` to `sigchld_handler()`.

