import numpy as np

from src.anomalies import (Anomaly, SphericalAnomaly, PrismAnomaly, CylindricalAnomaly,
                           ThinRodAnomaly, OkuboAnomaly, MogiAnomaly)
from src.geometry import Grid, Position, CoordinateList
from src.iterators import PropertyIterator, LinearPropertyIterator
from src.output import saveTimeseries, loadMetadata
import matplotlib.pyplot as plt
from src.model import Model
from src.cusum import CUSUM


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    def __main__
    Main entry point for the forward gravitry modeling scripts
    """

    heightmap = "plane"

    # Load station metadata (coordinates, ID) from disk
    x, y, names = loadMetadata("./data/stations.txt")
    # Get X, Y, Z receiver coordinates of stations
    receiver = CoordinateList(x, y, heightmap=heightmap)

    numtrials = 50
    ratedelay = np.empty((numtrials, 2))

    for itrial in range(numtrials):

        anomalysteps = 50 * (itrial + 1)

        quietperiod = 5000
        strength = 0.9 + np.random.rand()/2

        # Create a linear interpolator for two parameters: position & density over "N" steps
        operator = LinearPropertyIterator(
            anomalysteps, [
                ("normal", (0, strength)),
                ("deltadensity", (0, -200)),
            ]
        )

        # Linear interpolator for different properties: first one, and then the opposite for testing
        iterator = PropertyIterator([
            operator
        ])

        # OkuboAnomaly Parameters
        dip = 89.99
        strike = 90
        rake = 0
        length = 10000
        width = 10000
        slip = 0
        density = 2780
        fill = 2700

        # Save results for timeseries
        timeseries = list()

        for iteration in iterator:
            # source = OkuboAnomaly(Position(500076, 4176413, -3000), density, dip, strike, rake, length, width, slip,
            #                       iteration.normal, fill=fill)

            source = SphericalAnomaly(Position(500076, 4176413, -3000), iteration.deltadensity, 500)

            results = source.simulate(receiver)

            # Save the particular slice of time result
            timeseries.append(results)

            # Show plot of the field
            # plotGravityField(receiver, results)

        timeseries = np.array(timeseries)
        tsbase = np.zeros((quietperiod, timeseries.shape[1]))
        timeseries = np.append(tsbase, timeseries, axis=0)
        tspost = tsbase + np.mean(timeseries[-50:], axis=0)
        timeseries = np.append(timeseries, tspost, axis=0)

        # Noise should be added here when the time series are completed
        noiselevel = 0.03
        timeseries += np.random.normal(0, noiselevel, timeseries.shape)

        # ---------------------------------------------------------------------------------
        # Create a simple data model using a Pandas data frame
        model = Model(timeseries)
        # plotModel(model, window_length=1)

        # We can reuse the SimpleModel
        # Estimate mean for the simple model used in the CUSUM algo
        means, variances = model.simple(model="mean", window_length=100)

        # Get the upper and lower limits from the data frames
        high, low = CUSUM(model, means, variances, k=noiselevel)

        # Consider the maximum deviation between high and low
        maxdev = np.max([high, low], axis=0)

        # for all the stations
        maxdev = np.max(maxdev, axis=1)

        # set a dection threshold

        detecthres = 40

        detected = np.nonzero(maxdev > detecthres)

        if len(detected[0]) > 0:
            trigger = detected[0][0]
            delay = trigger - quietperiod
            if delay<0:
                delay = np.nan
        else:
            trigger = None
            delay = np.nan #'Not triggered'

        ratedelay[itrial] = [strength / anomalysteps, delay]


        # fig, (ax1, ax2) = plt.subplots(2)
        #
        # ax1.plot(maxdev, label="max CUSUM")
        # # ax1.axvline(x=quietperiod, linewidth=2, color='r')
        # # ax1.axvline(x=quietperiod+anomalysteps, linewidth=1, color='r', linestyle='--')
        # ax1.axvspan(quietperiod, quietperiod+anomalysteps,
        #             facecolor='coral', edgecolor="none",
        #             alpha=0.2, zorder=-10)
        # if trigger:
        #     ax1.axvline(x=trigger, linewidth=2, color='g')
        # ax1.set_title(f'Max CUSUM triggered with delay {delay} ')
        #
        # ax2.plot(timeseries)
        # ax2.set_title('Gravity')
        # ax2.set(ylabel='mgal')
        # ax2.axvspan(quietperiod, quietperiod+anomalysteps,
        #             facecolor='coral', edgecolor="none",
        #             alpha=0.2, zorder=-10)
        #
        # plt.show()
        #
        # exit()

marker_size = 15
plt.scatter(ratedelay[:, 0], ratedelay[:, 1], marker_size)

plt.title("CUSUM Anomaly Detection Sensitivity Test")
# plt.xlabel("opening rate [m/sample]")
plt.xlabel("density-contrast rate [kg/(m^3 * sample)]")
plt.ylabel("detection delay [samples]")

plt.show()
