#'Class to store timeseries objects
#'@param path to json
#'@return DataReader object
#'@examples
#'dr <- DataReader()
Rule <- function(time_series, indicator_series){
  object <- list(time_series=time_series, indicator_series=indicator_series)
  class(object) <-"Rule"
  return (object)
}

#' Generic Function dispatch
getTradeSignal <- function (x, ...) {
  UseMethod("getTradeSignal", x)
}


#'Create Trade signal given asset price series and indicator series
#'@param use_indicator boolean for whether to use indicator as trade signal or the underlying series as a trade signal
#'@return zoo
getTradeSignal.Rule <- function(x, buy_threshold, sell_threshold, hold_threshold=0,
                                use_indicator = TRUE) {
  if (use_indicator) {
    ts <- x$indicator_series
  } else {
    ts <- x$time_series[,'close']
  }
  ts_copy <- ts
  ts_copy[index(ts_copy)] <- 0
  ts_copy[index(ts[ts>sell_threshold])] <- (-1)
  ts_copy[index(ts[ts<buy_threshold])] <- 1
  ts_copy
}

