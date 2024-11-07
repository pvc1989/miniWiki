import sys
import numpy as np
from matplotlib import pyplot as plt
import argparse


def plot_mpi_balance(folder: str, n_core: int):
    values = np.zeros((n_core,))
    for i_core in range(n_core):
        with open(f'{folder}/log_{i_core}.txt', 'r') as file:
            line = file.readlines()[-1]
            pos = line.find('=') + 1
            values[i_core] = float(line[pos:])
            print(i_core, values[i_core])
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    ax.bar(range(n_core), values)
    plt.xlabel('comm rank')
    plt.ylabel('time cost')
    plt.tight_layout()
    plt.savefig(f'{folder}/balance_mpi.svg')


def plot_futures_balance(folder: str, n_task: int):
    process_to_tasks = dict()
    for i_task in range(n_task):
        with open(f'{folder}/log_{i_task}.txt', 'r') as file:
            lines = file.readlines()
            line = lines[-2]
            first = line.find('task') + 4
            last = line.find('on')
            task = int(line[first : last])
            first = line.find('process') + 7
            last = line.find('thread')
            process = int(line[first : last])
            first = last + 6
            last = -1
            thread = int(line[first : last])
            print(line, f'task = {task}, process = {process}, thread = {thread}')
            if process not in process_to_tasks:
                process_to_tasks[process] = list()
            line = lines[-1]
            first = line.find('=') + 1
            cost = float(line[first : last])
            process_to_tasks[process].append((i_task, cost))
    process_to_cost = dict()
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    for process, tasks in process_to_tasks.items():
        print('process', process)
        process_to_cost[process] = 0.0
        for task, cost in tasks:
            print('  task', task, 'costs', cost, 'sec')
            ax.bar(process, cost, bottom=process_to_cost[process])
            process_to_cost[process] += cost
    # plot the cost of each task as stacked bars
    plt.xlabel('process id')
    plt.ylabel('time cost')
    plt.tight_layout()
    plt.savefig(f'{folder}/balance_stacked.svg')
    # plot the cost of each process as a single bar
    plt.bar(process_to_cost.keys(), process_to_cost.values())
    plt.savefig(f'{folder}/balance_single.svg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Plot the work balance of each worker.')
    parser.add_argument('--folder', type=str, help='the folder contains the logs')
    parser.add_argument('--n_core', type=int, default=0, help='number of cores used in mpi')
    parser.add_argument('--n_task', type=int, default=0, help='number of tasks used in futures')
    args = parser.parse_args()

    folder = args.folder
    n_core = args.n_core
    n_task = args.n_task

    assert n_core == 0 or n_task == 0
    assert n_core != 0 or n_task != 0

    if n_core:
        plot_mpi_balance(folder, n_core)
    elif n_task:
        plot_futures_balance(folder, n_task)
    else:
        assert False
