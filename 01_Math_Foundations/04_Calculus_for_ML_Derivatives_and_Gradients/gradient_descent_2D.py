def f_2d(point):
    x, y = point
    return x**2 + y**2

def numerical_gradient(f, point, h=1e-7):
    gradient = []

    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)

        point_plus[i] += h
        point_minus[i] -= h

        partial = (f(point_plus) - f(point_minus)) / (2 * h)

        gradient.append(partial)

    return gradient

point = [4.0, 3.0]
lr = 0.1

for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
