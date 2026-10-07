import math
import random


def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorial(probs, n=1):
    cummulative = []
    total = 0

    for p in probs:
        total += p
        cummulative.append(total)

    samples = []
    for _ in range(n):
        r = random.random()

        for i, c in enumerate(cummulative):
            if r <= c:
                samples.append(i)
                break

    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []

    for _ in range(n):
        u1 = random.random()
        u2 = random.random()

        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)

    return samples
