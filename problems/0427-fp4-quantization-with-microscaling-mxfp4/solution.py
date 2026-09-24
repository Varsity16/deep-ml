import numpy as np

def mxfp4_quantize(x: list, block_size: int = 4) -> dict:
    """
    Perform MXFP4 quantization with per-block microscaling.
    """
    fp4_values = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=np.float32
    )

    x = np.asarray(x, dtype=np.float32)

    # Number of blocks
    num_blocks = (len(x) + block_size - 1) // block_size

    # Zero-pad if necessary
    padded_length = num_blocks * block_size
    padded = np.pad(
        x,
        (0, padded_length - len(x)),
        constant_values=0
    )

    # Reshape into blocks
    blocks = padded.reshape(num_blocks, block_size)

    quantized_blocks = []
    scales = []

    for block in blocks:
        # 1. Find maximum absolute value
        amax = np.max(np.abs(block))

        # 2. Calculate scale
        if amax == 0:
            scale = 1.0
        else:
            raw_scale = amax / 6.0

            # 3. Round UP to the nearest power of 2
            scale = 2 ** np.ceil(np.log2(raw_scale))

        scales.append(scale)

        # 4. Scale the values
        scaled = block / scale

        # 5. Find nearest FP4 value
        quantized = []

        for value in scaled:
            sign = np.sign(value)
            magnitude = abs(value)

            # Find distances from all positive FP4 values
            distances = np.abs(fp4_values - magnitude)

            # Minimum distance
            min_distance = np.min(distances)

            # Candidates in case of a tie
            candidates = fp4_values[distances == min_distance]

            # Tie → choose smaller absolute magnitude
            nearest = np.min(candidates)

            quantized.append(sign * nearest)

        quantized_blocks.append(quantized)

    # Flatten
    quantized_blocks = np.array(quantized_blocks).reshape(-1)

    # 6. Dequantize
    dequantized = np.empty_like(quantized_blocks)

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size

        dequantized[start:end] = (
            quantized_blocks[start:end] * scales[i]
        )

    # Remove padding
    dequantized = dequantized[:len(x)]

    return {
        "quantized": [round(float(v), 4) for v in dequantized],
        "scales": [round(float(s), 4) for s in scales]
    }