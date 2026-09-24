import numpy as np

def ptq(weights, calib_X):
    qmax = 127.0
    dequant_weights = {}

    # Quantize weights: per-output-channel INT8
    for name in ["W1", "W2", "W3"]:
        W = weights[name].astype(np.float32)

        max_abs = np.max(np.abs(W), axis=1)
        scales = np.where(max_abs == 0, 1.0, max_abs / qmax)

        Q = np.round(W / scales[:, None])
        Q = np.clip(Q, -qmax, qmax)

        dequant_weights[name] = (
            Q * scales[:, None]
        ).astype(np.float32)

    # Quantize biases too
    for name in ["b1", "b2", "b3"]:
        b = weights[name].astype(np.float32)

        max_abs = np.max(np.abs(b))
        scale = max_abs / qmax if max_abs != 0 else 1.0

        Q = np.round(b / scale)
        Q = np.clip(Q, -qmax, qmax)

        dequant_weights[name] = (
            Q * scale
        ).astype(np.float32)

    return dequant_weights