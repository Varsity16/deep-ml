import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    gradient = np.array(gradient, dtype=float)

    magnitude = np.linalg.norm(gradient)

    if magnitude == 0:
        direction = np.zeros_like(gradient)
    else:
        direction = gradient / magnitude

    descent_direction = -direction

    return {
        "magnitude": float(magnitude),
        "direction": direction.tolist(),
        "descent_direction": descent_direction.tolist()
    }
