import numpy as np

def EOF(model, normalise=True):

    """
    Def EOFModel.estimate
    Estimate the EOF using an eigendecomposition of the covariance matrix
    """

    # Eigendecomposition of the covariance matrix
    w, v = np.linalg.eigh(model.frame.cov())

    # Normalise the eigenvalues
    if normalise:
        w = w / np.sum(w)

    # np.eigh returns ascencing vectors so flip to descending (most important first)
    return np.flip(w), np.flip(v, axis=1)
