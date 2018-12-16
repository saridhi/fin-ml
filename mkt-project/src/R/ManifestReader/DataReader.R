library(ini)

#'Read data from a specified JSON
#'@param path to json
#'@return DataReader object
#'@examples
#'dr <- DataReader()
DataReader <- function(path = "/Users/dhirensarin/Dev/project-ada/data/configs/wiki_eod_short.json"){
  json_data <- fromJSON(file=path)
  class(json_data) <- "DataReader"
  return (json_data)
}

#' Generic Function dispatch
getTimeSeries <- function (x, ...) {
  UseMethod("getTimeSeries", x)
}

#' Generic Function dispatch
listTickers <- function (x, ...) {
  UseMethod("listTickers", x)
}

#' Read CSV and store in memory
readCsv.uncached <- function(x) {
  prop_contents <- ini::read.ini('../../properties/properties.cfg')
  csv_names <- paste0(prop_contents$`Data Paths`$base_path, x$relative_path)
  csv_contents <- NULL
  
  for (csv_name in csv_names) {
    if ((file.info(csv_name)$size) > 5000000) {
      csv_contents <- rbind(csv_contents, fread(file=csv_name, header = TRUE))
    } else {
      csv_contents <- rbind(csv_contents, read.csv(csv_name))
    }
  }
  csv_contents <- data.frame(csv_contents)
  columns <- sub(' ', '.', x$attribute_columns)
  if (is.null(x$ticker_column)) {
    csv_contents <- csv_contents[,c(x$date_column, columns)]
  } else {
    ticker <- x$ticker_column
    csv_contents <- csv_contents[,c(x$date_column, ticker, columns)]
  }
  
  #csv_contents <- csv_contents[order(x$date_column),] 
  
  if ((x$rename_map[1]) == 'toLowerCase') {
    colnames(csv_contents) <- c(x$date_column, tolower(columns))
  }
  else {
    renamed_fields <- sub(' ', '.',names(x$rename_map))
    colnames(csv_contents)[colnames(csv_contents) %in% renamed_fields] <- as.character(x$rename_map)
  }
  csv_contents
}
if (!exists("readCsv")) readCsv <- memoise(readCsv.uncached) 

#'Fetch specific ticker history from flat file
#'@return zoo
getTimeSeries.DataReader <- function(x, ticker = "A", fields = c('open', 'high', 'low', 'close', 'volume'), 
                                     start_date = as.Date('2005-01-01'), 
                                     end_date = Sys.Date(), 
                                     frequency = x$frequency) {
  csv_contents <- readCsv(x)
  if (is.null(ticker)) {
    df <- csv_contents[, c(x$date_column, fields)]
  } else {
    df <- tryCatch({
      csv_contents[csv_contents[x$ticker_column] == ticker, c(x$date_column, fields)]
    }, error = function(e) {
      csv_contents[, c(x$date_column, fields)]
    })
  }
  numeric.series <- apply(df[,2:NCOL(df)], 2, as.numeric)
  
  #Check date format
  if (sum(is.na(as.Date(df[,1]))) > 100) {
    df[,1] <- gsub("[0-9]{2}([0-9]{2})$", "\\1", df[,1]) 
    dates <- as.Date(df[,1], format = "%m/%d/%y")
  } else {
    dates <- as.Date(df[,1])
  }
  
  series.zoo <- zoo(numeric.series, order.by = dates)
  window(series.zoo, start=start_date, end=end_date)
}

#'List all unique tickers from flat file
listTickers.DataReader <- function(x) {
  csv_contents <- readCsv(x)
  if (is.null(x$ticker_column)) {
    return (x$raw_data_id)
  } else {
    return (unique(csv_contents[x$ticker_column]))
  }
}

