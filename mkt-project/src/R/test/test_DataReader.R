library(rjson)
library(data.table)
library(memoise)
library(zoo)
library(magrittr)
library(quantmod)
library(ini)
source('ManifestReader/DataReader.R')
source('ManifestReader/ManifestReader.R')
source('FeatureEngineer/Rule.R')

manifest_reader <- ManifestReader()
stocks_reader <- getDataReader(manifest_reader, "quandl:wiki/eod_short")
vix_reader <- getDataReader(manifest_reader, "cboe:vix")

vix_reader %>% listTickers()
nflx_series <- getTimeSeries(stocks_reader, ticker = 'NFLX')
vix_series <- getTimeSeries(vix_reader, ticker = 'vix', fields = c('open', 'high', 'low', 'close'))

snp_reader <- getDataReader(manifest_data, "yahoo:snp_500")
snp_series <- getTimeSeries(snp_reader, ticker = NULL)

#rsi_series can act as a feature. 
#rsi_series values example: 32.15, 33.25, 60.20 etc
rsi_series <- RSI(nflx_series[,4], 9)
rsi_rule <- Rule(nflx_series, rsi_series)
#rsi_trade_signal can act as a feature. This is categorical rather than continuous.
#rsi_trade_signal values example: 1, 1, 1, 0, -1, -1, 1 etc.
rsi_trade_signal <- getTradeSignal(rsi_rule, buy_threshold=20, sell_threshold=80)

#ma_series - Moving average can act as a feature
ma_series <- SMA(nflx_series[,'close'], 21)
ma_rule <- Rule(nflx_series, ma_series)
ma_trade_signal <- getTradeSignal(ma_rule, buy_threshold=ma_series, sell_threshold=ma_series)

bb_series <- BBands(nflx_series[,c('high', 'low', 'close')], n=21)
bb_rule <- Rule(nflx_series, bb_series)
bb_trade_signal <- getTradeSignal(bb_rule, buy_threshold = bb_series$dn, sell_threshold=bb_series$up,
                                  use_indicator = FALSE)



backtest_rsi <- Backtest(nflx_series$close, rsi_trade_signal)
backtest_result <- backtest_rsi %>% doBacktest()

dygraph(cumsum(backtest_result$returns))
