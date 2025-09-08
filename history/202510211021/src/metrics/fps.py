import time

class FpsMeter:
    def __init__(self):
        self.start = time.time()
        self.frames = 0

    def update(self):
        self.frames += 1
        elapsed = max(time.time() - self.start, 0.001)
        return self.frames / elapsed
