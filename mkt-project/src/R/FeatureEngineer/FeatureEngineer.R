#'Library of functions that are relevant for Feature generation and engineering

#'Create Features for a time series
#'@param time_series 
#'@return zoo
create_features <- function (time_series, feature_reader) {
  
  features <- getColumns(feature_reader)
  feature_names <- names(features)
  
  features_list <- list()
  
  for (feature_name in feature_names) {
    feature_attribs <- features[[feature_name]]
    feature_series <- switch(feature_attribs$feature_type,
                             'RSI_rule' = {
                               rsi_series <- RSI(time_series$close, feature_attribs$period)
                               rsi_rule = Rule(time_series, rsi_series)
                               getTradeSignal(rsi_rule, buy_threshold = feature_attribs$oversold, 
                                              sell_threshold = feature_attribs$overbought)
                             },
                             'RSI' = RSI(time_series$close, feature_attribs$period),
                             'SMA_rule' = {
                               ma_series <- SMA(time_series$close, feature_attribs$period)
                               ma_rule <- Rule(time_series, ma_series)
                               getTradeSignal(ma_rule, 
                                              buy_threshold = ma_series, 
                                              sell_threshold = ma_series,
                                              use_indicator = FALSE)
                             },
                             'ADX' = {
                               adx_series <- ADX(c(time_series[,c('high', 'low', 'close')]), 
                                                 feature_attribs$period)[,4]
                             },
                             'ADX_rule' = {
                               adx_series <- ADX(as.xts(c(time_series[,c('high', 'low', 'close')])), 
                                                 feature_attribs$period)[,4]
                               adx_rule <- Rule(time_series, adx_series)
                               getTradeSignal(adx_rule, 
                                              buy_threshold = feature_attribs$trending, 
                                              sell_threshold = feature_attribs$trending,
                                              use_indicator = TRUE)
                             },
                             'Volatility' = {
                               volatility(time_series$close, calc='close', N=feature_attribs$period)
                             },
                             'BB_rule' = {
                               bb_series <- BBands(time_series$close, n=feature_attribs$period, sd=feature_attribs$sd)
                               bb_rule <- Rule(time_series, bb_series)
                               getTradeSignal(bb_rule, 
                                              buy_threshold = bb_series$dn, 
                                              sell_threshold = bb_series$up,
                                              use_indicator = FALSE)
                             }
    )
    features_list <- c(features_list, list(feature_series))
  }
  features_zoo <- do.call(merge, features_list)
  colnames(features_zoo) <- feature_names
  features_zoo
}

#'Create Targets for a time series
#'@param time_series 
#'@param lag up/down n periods look forward
#'@param use_sign if TRUE return a +1, -1 else return the percent difference
#'@return zoo
create_targets <- function(time_series, lag=-1, use_sign=TRUE) {
  percent_diffs <- Delt(time_series$close)
  if (use_sign) {
    sign(lag(percent_diffs, lag))
  } else {
    lag(percent_diffs, lag)
  }
}
