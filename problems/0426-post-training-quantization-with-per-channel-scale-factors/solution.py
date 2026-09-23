import numpy as np

def per_channel_quantize(weight: np.ndarray, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.
    """
    qmax = 2**(bits - 1) - 1

    # Maximum absolute value for each output channel (row)
    max_abs = np.max(np.abs(weight), axis=1)

    # Calculate one scale per channel
    scale_factors = max_abs / qmax

    # Avoid division by zero for channels containing only zeros
    scale_factors = np.where(scale_factors == 0, 1.0, scale_factors)

    # Quantize each row using its own scale
    quantized_weights = np.round(
        weight / scale_factors[:, np.newaxis]
    ).astype(np.int32)

    # Dequantize
    dequantized_weights = (
        quantized_weights * scale_factors[:, np.newaxis]
    )

    return quantized_weights, scale_factors, dequantized_weights