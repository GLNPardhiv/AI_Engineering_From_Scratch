import random

from mini_mlp_from_scratch import MLP

random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) # type: ignore

    for p in model.parameters():
        p.grad = 0.0

    loss.backward() # type: ignore

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}") # type: ignore

if __name__ == "__main__":
    print("\nPredictions after training:")
    
    for x, y in zip(xs, ys):
        print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}") # type: ignore
