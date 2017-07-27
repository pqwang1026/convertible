import utils.perf as perf

@perf.timed
def test(x):
    return x + 1

test(3)