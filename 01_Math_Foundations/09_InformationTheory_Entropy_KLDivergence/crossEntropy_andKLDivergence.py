import math

from informationContent_and_entropy import entropy


def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))

    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

if __name__ == "__main__":
    true_dist = [0.7, 0.2, 0.1]
    good_model = [0.6, 0.25, 0.15]
    bad_model = [0.1, 0.1, 0.8]

    print(f"Entropy of true dist:     {entropy(true_dist):.4f} bits")
    print(f"CE (good model):          {cross_entropy(true_dist, good_model):.4f} bits")
    print(f"CE (bad model):           {cross_entropy(true_dist, bad_model):.4f} bits")
    print(f"KL divergence (good):     {kl_divergence(true_dist, good_model):.4f} bits")
    print(f"KL divergence (bad):      {kl_divergence(true_dist, bad_model):.4f} bits")
