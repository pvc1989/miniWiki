import time
import argparse
import sys
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, wait


def one_task(i_task: int) -> int:
    time.sleep(1)
    sys.stdout.write(f'thread {threading.current_thread().ident}, task {i_task}\n')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}.py',
        description = 'Demo the basic usage of the concurrent.futures module.')
    parser.add_argument('--n_thread',  type=int, help='number of threads for running futures')
    parser.add_argument('--n_task',  type=int, help='number of tasks for running futures, usually >> n_thread')
    args = parser.parse_args()

    run_tasks_by_threads(args.n_thread, args.n_task)
    print(args)
