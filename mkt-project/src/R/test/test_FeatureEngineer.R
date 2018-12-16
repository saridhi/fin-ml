library(rjson)
library(data.table)
library(memoise)
library(zoo)
library(magrittr)
library(quantmod)
library(dygraphs)
library(ini)
library(PerformanceAnalytics)
library(caret)
source('ManifestReader/DataReader.R')
source('ManifestReader/ManifestReader.R')
source('ManifestReader/FeatureReader.R')
source('MLAlgorithms/KNN_Algo.R')
source('FeatureEngineer/Rule.R')
source('FeatureEngineer/FeatureEngineer.R')
source('Backtest.R')

manifest_data <- ManifestReader(path = "/Users/dhirensarin/Dev/project-ada/data/configs")
manifest_features <- ManifestReader(path = "/Users/dhirensarin/Dev/project-ada/features/configs")
knn_features <- getFeatureReader(manifest_features, 'knn_1')

data_reader <- getDataReader(manifest_data, "quandl:wiki/eod_short")
snp_reader <- getDataReader(manifest_data, "yahoo:snp_500")
vix_reader <- getDataReader(manifest_reader, "cboe:vix")

vix_series <- getTimeSeries(vix_reader, ticker = 'vix', fields = c('open', 'high', 'low', 'close'))
time_series <- getTimeSeries(data_reader, ticker = 'NFLX')
snp_series <- getTimeSeries(snp_reader, ticker = NULL)

'The function call below returns a time series like:
            RSI_rule  RSI       SMA_rule
2017-05-16        -1 65.15046       1
2017-05-17        0 43.61202        1
2017-05-18        0 50.95530        -1 
'
features <- create_features(time_series, feature_reader = knn_features)
targets <- create_targets(time_series)

data_set <- na.omit(na.locf(merge(features, targets)))
knn_predictions <- doKnn(data_set, retrain_gap=500)
backtest_knn <- Backtest(time_series$close, knn_predictions)
backtest_result <- backtest_knn %>% doBacktest()

dygraph(merge(cumsum(backtest_result$returns), cumsum(na.fill(Delt(time_series$close), 0))))
dygraph(cumsum(backtest_result$returns))
