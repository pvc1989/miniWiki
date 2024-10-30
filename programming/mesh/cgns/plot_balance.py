import sys
import numpy as np
from matplotlib import pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}.py',
        description = 'Plot the work balance of each worker.')
    parser.add_argument('--folder', type=str, help='the folder contains the logs')
    parser.add_argument('--n_task', type=int, help='number of tasks')
    args = parser.parse_args()

    folder = args.folder
    n_task = args.n_task
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
