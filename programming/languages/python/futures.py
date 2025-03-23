import time
import argparse
import sys
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait


global_matrix = np.ndarray((3,3))


def task_by_one_worker(i_task: int, matrix: np.ndarray) -> int:
    norm = 0.0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            norm += matrix[i][j]**2
    norm = np.sqrt(norm)
    # Effectively np.linalg.norm(matrix), but intentionally made slow.
    sys.stdout.write(f'norm = {norm:.4e} from task {i_task} on process {os.getpid()} thread {threading.current_thread().ident}\n')
    return i_task


def task_by_one_thread(i_task: int, shared_matrix: np.ndarray) -> int:
    return task_by_one_worker(i_task, shared_matrix)


def task_by_one_process(i_task: int) -> int:
    global global_matrix
    return task_by_one_worker(i_task, global_matrix)


def run_tasks_by_threads(n_thread, n_task, n_scalar):
    shared_matrix = np.random.rand(n_scalar, n_scalar)
    # heap data automatically shared by all threads in the same process
    executor = ThreadPoolExecutor(max_workers=n_thread)
    start = time.time()
    futures = []
    for i_task in range(n_task):
        futures.append(executor.submit(task_by_one_thread, i_task, shared_matrix))
    wait(futures)
    end = time.time()
    duration = end - start
    print(f'total cost: {duration:.2f}s')
    print(f"norm = {np.linalg.norm(shared_matrix):.4e} from the main thread")
    i_task_sum = 0
    for future in futures:
        i_task_sum += future.result()
    print(f"i_task_sum from all threads = {i_task_sum}")
    i_task_sum = np.sum(np.arange(n_task))
    print(f"i_task_sum from the main thread = {i_task_sum}")
    executor.shutdown()


def run_tasks_by_processes(n_process, n_task, n_scalar):
    global global_matrix
    global_matrix = np.random.rand(n_scalar, n_scalar)
    # global data automatically shared by all children of the same process
    executor = ProcessPoolExecutor(max_workers=n_process)
    start = time.time()
    futures = []
    for i_task in range(n_task):
        futures.append(executor.submit(task_by_one_process, i_task))
    wait(futures)
    end = time.time()
    duration = end - start
    print(f'total cost: {duration:.2f}s')
    print(f"norm = {np.linalg.norm(global_matrix):.4e} from the main process")
    i_task_sum = 0
    for future in futures:
        i_task_sum += future.result()
    print(f"i_task_sum from all processes = {i_task_sum}")
    i_task_sum = np.sum(np.arange(n_task))
    print(f"i_task_sum from the main process = {i_task_sum}")
    executor.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Demo the basic usage of the concurrent.futures module.')
    parser.add_argument('--n_thread',  type=int, default=0, help='number of threads for running futures')
    parser.add_argument('--n_process',  type=int, default=0, help='number of processes for running futures')
    parser.add_argument('--n_task',  type=int, help='number of tasks for running futures, usually >> n_thread')
    parser.add_argument('--n_scalar',  type=int, default=10000, help='number of scalars in the vector')
    args = parser.parse_args()

    np.random.seed(123456789)
    if args.n_thread:
        run_tasks_by_threads(args.n_thread, args.n_task, args.n_scalar)
    elif args.n_process:
        run_tasks_by_processes(args.n_process, args.n_task, args.n_scalar)
    print(args)
