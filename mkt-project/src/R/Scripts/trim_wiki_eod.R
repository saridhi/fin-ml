
all_ts <- lapply(as.vector(trim_tickers), function(x) {
  print(x)
  tryCatch({
    getTimeSeries(dr, ticker=x)
  }, error = function(e) {
    return (NULL)
  })
})

dates <- index(ts)

#Script to trim wiki_eod.csv
ticker_list <- lapply(dates, function(x) {
  all_volumes <- csv_contents[csv_contents$date==as.character(x),"adj_volume"]
  all_tickers <- csv_contents[csv_contents$date==as.character(x),"ticker"]
  ticker_index <- which(all_volumes %in% sort(as.numeric(all_volumes), decreasing = TRUE)[1:10])
  print(x)
  all_tickers[ticker_index]
})

series.volumes <- lapply()
