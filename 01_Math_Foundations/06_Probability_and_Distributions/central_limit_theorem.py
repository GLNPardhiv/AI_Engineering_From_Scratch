def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []

    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))

    return averages
