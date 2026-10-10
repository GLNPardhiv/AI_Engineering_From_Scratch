import math


def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

if __name__ == "__main__":
    fair_coin = [0.5, 0.5]
    biased_coin = [0.99, 0.01]
    fair_die = [1/6] * 6

    print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
    print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
    print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")
