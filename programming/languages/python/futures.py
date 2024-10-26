import time
import argparse
import sys
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait


def one_task(i_task: int) -> int:
    time.sleep(1)
    sys.stdout.write(f'task {i_task} on process {os.getpid()} thread {threading.current_thread().ident}\n')
    return i_task


def run_tasks_by_threads(n_thread, n_task):
    executor = ThreadPoolExecutor(n_thread)
    start = time.time()
    futures = []
    for i_task in range(n_task):
        futures.append(executor.submit(one_task, i_task))
    wait(futures)
    end = time.time()
    duration = end - start
    print(f'total cost: {duration:.2f}s')
    i_task_sum = 0
    for future in futures:
        i_task_sum += future.result()
    print(f"i_task_sum from all threads = {i_task_sum}")
    i_task_sum = np.sum(np.arange(n_task))
    print(f"i_task_sum from main thread = {i_task_sum}")
    executor.shutdown()


def run_tasks_by_processes(n_process, n_task):
    executor = ProcessPoolExecutor(n_process)
    start = time.time()
    futures = []
    for i_task in range(n_task):
        futures.append(executor.submit(one_task, i_task))
    wait(futures)
    end = time.time()
    duration = end - start
    print(f'total cost: {duration:.2f}s')
    i_task_sum = 0
    for future in futures:
        i_task_sum += future.result()
    print(f"i_task_sum from all processes = {i_task_sum}")
    i_task_sum = np.sum(np.arange(n_task))
    print(f"i_task_sum from main process = {i_task_sum}")
    executor.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}.py',
        description = 'Demo the basic usage of the concurrent.futures module.')
    parser.add_argument('--n_thread',  type=int, default=0, help='number of threads for running futures')
    parser.add_argument('--n_process',  type=int, default=0, help='number of processes for running futures')
    parser.add_argument('--n_task',  type=int, help='number of tasks for running futures, usually >> n_thread')
    args = parser.parse_args()

    if args.n_thread:
        run_tasks_by_threads(args.n_thread, args.n_task)
    elif args.n_process:
        run_tasks_by_processes(args.n_process, args.n_task)
    print(args)
