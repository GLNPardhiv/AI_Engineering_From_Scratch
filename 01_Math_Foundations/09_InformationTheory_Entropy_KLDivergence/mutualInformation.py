import math


def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi

if __name__ == "__main__":
    independent = [[0.25, 0.25], [0.25, 0.25]]
    dependent = [[0.45, 0.05], [0.05, 0.45]]

    print(f"MI (independent): {mutual_information(independent):.4f} bits")
    print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")
