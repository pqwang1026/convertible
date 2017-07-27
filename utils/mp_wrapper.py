import multiprocessing as mp
import logging
from utils.perf import Timer

logger = logging.getLogger(__name__)


def f(x, y, z=2):
    return x * y * z


def run(f, args, processes=mp.cpu_count() - 2):
    timer = Timer()
    logger.info('Multiprocessing {0} in {1} tasks...'.format(f, len(args)))

    timer.start()

    res_list = []

    def log_result(res):
        res_list.append(res)

    pool = mp.Pool(processes=processes)
    for arg in args:
        pool.apply_async(f, kwds=arg, callback=log_result)
    pool.close()
    pool.join()

    logger.info('Multiprocessing finished.')
    timer.stop()

    return res_list


if __name__ == '__main__':
    args = [{'x': 1, 'y': 2, 'z': 3}, {'x': 2, 'y': 2, 'z': 3}]

    res = run(f, args)
    print(res)
