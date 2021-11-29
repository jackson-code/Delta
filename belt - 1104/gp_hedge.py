import math

def compute_probabilities(gains, eta):
    total = sum([math.exp(eta * gain) for gain in gains])
    return [math.exp(eta * gain) / total for gain in gains]

def choose_x(max_points, p):
    chosen_x = None
    chosen_idx = None
    max_probability = -1
    for idx in range(0, len(p)):
        if max_probability < p[idx]:
            max_probability = p[idx]
            chosen_x = max_points[idx]
            chosen_idx = idx
    return chosen_x, chosen_idx