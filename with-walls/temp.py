from multiprocessing.dummy import Pool
from multiprocessing import cpu_count

class Temp:
    def __init__(self):
        self.pool = Pool(cpu_count())
        self.a = 1

    def temp(self):
        def test_f(i):
            return i
        self.pool.map(test_f, range(1000))


if __name__ == "__main__":
    t = Temp()
    t.temp()
