def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll.
    """
    expected_value = (n + 1) / 2
    variance = (n ** 2 - 1) / 12

    return expected_value, variance