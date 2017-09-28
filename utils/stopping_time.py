import numpy as np
import pandas as pd
import logging
import utils.perf as perf


class Path(tuple):
    UP = 1
    DOWN = -1

    def __new__(cls, a, ):
        return super(Path, cls).__new__(cls, tuple(a))

    def __add__(self, other):
        return Path(list(self) + list(other))


class StoppingTime(dict):
    def __init__(self, profile: dict):
        for path, time in profile.items():
            self[path] = time


def get_all_paths(num_steps):
    if num_steps == 1:
        return [Path([1]), Path([-1])]
    later_paths = get_all_paths(num_steps - 1)
    first = [Path([1]) + p for p in later_paths]
    second = [Path([-1]) + p for p in later_paths]
    return first + second


def get_all_stopping_times(num_steps):
    if num_steps == 1:
        return [StoppingTime({Path([1]): 0, Path([-1]): 0, }),
                StoppingTime({Path([1]): 1, Path([-1]): 1, })]
    later_st = get_all_stopping_times(num_steps - 1)
    zero_st = StoppingTime({path: 0 for path in get_all_paths(num_steps)})

    all_st = [zero_st]
    for up_st in later_st:
        for dn_st in later_st:
            up_st_new = StoppingTime({Path([1]) + path: time + 1 for path, time in up_st.items()})
            dn_st_new = StoppingTime({Path([-1]) + path: time + 1 for path, time in dn_st.items()})
            all_st.append(StoppingTime({**dict(up_st_new), **dict(dn_st_new)}))
    return all_st


if __name__ == '__main__':
    import pprint

    pprint.pprint(get_all_stopping_times(3), width=1)
