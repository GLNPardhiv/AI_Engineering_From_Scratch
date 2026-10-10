import math
import random

from crossEntropy_as_classificationLoss import (  # type: ignore
    cross_entropy_loss,
    softmax,
)

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

if __name__ == "__main__":
    print(f"Cross-entropy loss:      {ce_loss:.6f}")
    print(f"Negative log-likelihood: {nll:.6f}")
    print(f"Difference:              {abs(ce_loss - nll):.2e}")
