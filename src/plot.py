import numpy as np
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy.io import loadmat

def plotCUSUMGraph(high, low, mode="all"):

    """
    Def plotCUSUMGraph
    Decides what CUSUM parameter to plot based on multiple traces
    """

    fig, (ax1, ax2) = plt.subplots(2)

    if mode == "all":
        ax1.plot(high, label="Upper")
        ax2.plot(low, label="Lower")
    elif mode == "median":
        ax1.plot(np.median(high, axis=1), label="Upper")
        ax2.plot(np.median(low, axis=1), label="Lower")
    elif mode == "mean":
        ax1.plot(np.mean(high, axis=1), label="Upper")
        ax2.plot(np.mean(low, axis=1), label="Lower")
    elif mode == "max":
        ax1.plot(np.max(high, axis=1), label="Upper")
        ax2.plot(np.max(low, axis=1), label="Lower")
    else:
        raise ValueError("Unknown mode type requested.")

    fig.suptitle("CUSUM Change Detection")

    # Show legend when not plotting all gravimeters
    if mode != "all":
        ax1.legend()
        ax2.legend()

    plt.show()


def plotEOFEigenvalues(eig):

    """
    def plotEigenvalues
    Plots relative importance of eigenvalues
    """

    plt.title("EOF mode weights (eigenvalues)")
    plt.semilogy(eig)
    plt.scatter(np.arange(eig.size), eig)
    plt.show()


def plotHeightmapContours():

    """
    Def plotHeightmapContours
    Plots Mt. Etna heightmap contours sourced from MatLab file
    """

    # Etna heightmap
    matlabObject = loadmat("./data/etna.mat")

    E = matlabObject["E1"][0,:]
    N = matlabObject["N1"][:,0]
    H = matlabObject["H1"]

    plt.contour(E, N, H, 25, linewidths=0.5, cmap=cm.gray)


def plotEOFEigenvectors(eigenvalues, eigenvectors, model, weight_cutoff=0.05):

    """
    def plotEOFEigenvectors
    Plots eigenvectors on spatial map using station coordinates
    """
    
    # Eigenvectors are in columns: transpose
    for weight, vector in zip(eigenvalues, eigenvectors.T):

        if weight < weight_cutoff:
            break

        # Project the eigenvector back
        plt.subplot(2, 1, 1)
        plt.suptitle("EOF & PCA with weight %.5f" % weight)
        plt.tricontourf(*model.coordinates.T, vector, 20)
        plt.colorbar(orientation="horizontal")

        # Etna height map
        plotHeightmapContours()

        # Plot and annotate station locations
        plt.scatter(*model.coordinates.T, color="red")

        # Annotate
        for coordinate, txt in zip(model.coordinates, model.frame.columns):
            text = plt.gca().annotate(txt, coordinate, color="white", ha="center", va="center", size=8)
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground="black"),
                                   path_effects.Normal()])

        # Coefficient is data projected on the eigenvector
        plt.subplot(2, 1, 2)
        plt.plot(model.values @ vector)

        plt.show()


def plotQuantile(mean, quantile, color):

    """
    Def Model.plotQuantile
    Plots particular quantile for data in the model
    """

    plt.fill_between(
        np.arange(len(mean)),
        mean.quantile(quantile, axis=1).values,
        mean.quantile(1 - quantile, axis=1).values,
        color=color
    )

def plotModel(model, mode="all", window_length=100):

    mean = model.frame.rolling(window_length).mean()

    if mode == "median":
        mean.mean(axis=1).plot()
    elif mode == "all":
        mean.plot()
    elif mode == "stack":
        mean.sum(axis=1).plot()
    elif mode == "mean":
        mean.mean(axis=1).plot()
    elif mode == "quantile":
        plotQuantile(mean, 0.021, "lightgray")
        plotQuantile(mean, 0.136, "darkgray")
        plotQuantile(mean, 0.341, "gray")
        plt.plot(mean.median(axis=1).values, color="black")
    else:
        raise ValueError("Unknown mode type requested.")

    plt.title("Gravimeter data (%s)" % mode)
    plt.show()
