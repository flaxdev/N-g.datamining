import numpy as np

def autoRegressionModel(offset, sigma, coefficients, shape):

  """
  def autoRegressionModel
  Based on https://en.wikipedia.org/wiki/Autoregressive_model#Definition
  Simulates auto-regressive noise
  """

  # Offset is the "c" parameter
  # Coefficients is phi_i
  coefficients = coefficients
  order = len(coefficients)

  # Create white noise (epsilon_t)
  noise = np.random.normal(0, sigma, shape)

  # Recursive fill of the first number of "order" values
  # When the order is 3, only the first element is drawn. Then an AR(1) is used
  # to calculate the second element, then a AR(2) is applied to calculate the third term. Finally,
  # the AR(3) is used to calculate the full sequence
  if order > 1:
    # Decrement the order
    values = autoRegressionModel(offset, sigma, coefficients[1:], (order, shape[1]))
  else:
    values = [offset + noise[0]]

  # Create the requested number of values over the timeseries
  for t in range(order, shape[0]):

    # Offset is included in all values
    value = offset

    # Go over all coefficients
    for i, coefficient in enumerate(coefficients):
      value += values[t - (i + 1)] * coefficient

    values.append(value + noise[t])

  return values
