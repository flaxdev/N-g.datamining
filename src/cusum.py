import numpy as np

def CUSUM(model, means, variances, k=1):

    """
    Def cusum
    Vectorised CUSUM algorithm implementation (on multiple traces at once)
    """

    # Normalise the values (CUSUM algo) by subtracting & dividing data frames
    normalised = (model.frame - means) / np.sqrt(variances)

    # Containers for CHigh and CLow
    high, low = np.zeros(normalised.shape), np.zeros(normalised.shape)

    # Mask for nan values
    mask = np.zeros(model.frame.columns.shape)

    # CUSUM algorithm
    for i, value in enumerate(normalised.values):

        # Zero makes no sense (Python assumes that [-1] is the last element of the array)
        if i == 0:
          continue

        # Element-wise maximum/minimum where (ignore NaN)
        high[i] = np.fmax(mask, high[i - 1] + value - k)
        low[i] = np.fmax(mask, low[i - 1] - value - k)

    return high, low
