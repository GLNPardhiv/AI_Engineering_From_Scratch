from micrograd_from_scratch import Value


def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)

    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff

if __name__ == "__main__":
    def expr(x):
        return (x ** 3 + x * 2 + 1).tanh()

    ad, num, diff = gradient_check(expr, 0.5)
    print(f"Autodiff:  {ad:.8f}")
    print(f"Numerical: {num:.8f}")
    print(f"Difference: {diff:.2e}")
    # Difference should be < 1e-5
