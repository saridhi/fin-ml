library(quantmod)

#'Library of relevant indicators to build features for the ML algorithms.
#'These are lightweight helper functions.

#'Generic Function to calculate percent distance between indicator and price
#'@param type 'Percent' or 'Price'
#'@return zoo
ma_distance_from_px <- function (time_series, ma_series, type='Percent') {
  if (type=='Percent') {
    (time_series-ma_series)/ma_series
  } else {
    time_series-ma_series
  }
}
