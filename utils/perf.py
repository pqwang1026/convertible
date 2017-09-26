import time
import logging

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self, name='', logger=logger, func=time.perf_counter):
        self.logger = logger
        self._name = name
        self._elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self._elapsed += end - self._start
        self._start = None

        if self.logger is not None:
            self.logger.debug('{} elapsed: {:.6f} secs'.format(self._name, round(self._elapsed, 6)))

    def reset(self):
        self._elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    @property
    def elapsed(self):
        return self._elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def timed(function):
    def wrapper(*args, **kwargs):
        with Timer('function [{0}]'.format(function.__name__), logger=logger):
            res = function(*args, **kwargs)
        return res

    return wrapper


if __name__ == '__main__':
    pass
