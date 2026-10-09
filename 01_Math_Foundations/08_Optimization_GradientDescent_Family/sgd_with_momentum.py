class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)

        self.velocity = [self.momentum * v + g for v, g in zip(self.velocity, grads)]

        return [p - self.lr * v for p, v in zip(params, self.velocity)]
