import numpy as np


def get_range(i_task: int, n_task: int, n_global: int):
    assert 0 <= i_task < n_task
    n_local_upper = (n_global + n_task - 1) // n_task
    n_local_lower = n_global // n_task
    n_upper = n_global % n_task

    if i_task < n_upper:
      first = n_local_upper * i_task
      last = first + n_local_upper
    else:
      first = n_local_upper * n_upper + n_local_lower * (i_task - n_upper)
      last = first + n_local_lower
    return first, last


def _test(n_task: int, n_global: int):
    n_local_all = np.ndarray(n_task, dtype=int)
    for i_task in range(n_task):
       first, last = get_range(i_task, n_task, n_global)
       n_local_all[i_task] = last - first
    assert np.sum(n_local_all) == n_global
    n_local_lower = n_global // n_task
    assert (n_local_lower <= n_local_all).all()
    n_local_upper = (n_global + n_task - 1) // n_task
    assert (n_local_all <= n_local_upper).all()


if __name__ == "__main__":
    n_global = 1025
    for n_task in range(1, n_global // 2):
        _test(n_task, n_global)
