from adam import Adam
from sgd_with_momentum import SGDMomentum  # type: ignore
from vanilla_gradient_descent import GradientDescent  # type: ignore


def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)

    return [df_dx, df_dy]

def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]

    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])

    return history

if __name__ == "__main__":
    start = [-1.0, 1.0]

    gd_history = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start)
    sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)
    adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)

    for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("Adam", adam_history)]:
        final = history[-1]
        loss = rosenbrock(final)
        print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")
