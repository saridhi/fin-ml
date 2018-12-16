#'Class to backtest given a time_series and trade signals
#'@param closes of asset
#'@param trade_signals timeseries of buy/sell signals 
#'@return Backtest object
#'@examples
Backtest <- function(closes, trade_signals){
  object <- list(closes=closes, trade_signals=trade_signals)
  class(object) <-"Backtest"
  return (object)
}

#' Generic Function dispatch
doBacktest <- function (x, ...) {
  UseMethod("doBacktest", x)
}


#'Perform backtest
#'@param risk_manager object that contains position sizes and logic for stoplosses/take profit
#'@return performance object containing information on strategy performance
doBacktest.Backtest <- function(x, risk_manager=NULL, return_type='Percent') {
  changes <- Delt(x$closes)
  signals <- lag(x$trade_signals, -1)
  list(returns = signals * changes,
       no_of_trades = sum(diff(x$trade_signals)!=0),
       sharpe = SharpeRatio(na.omit(signals * changes)))
}

