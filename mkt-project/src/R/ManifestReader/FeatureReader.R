library(ini)

#'Read features from a specified JSON
#'@param path to json
#'@return DataReader object
#'@examples
#'dr <- DataReader()
FeatureReader <- function(path = "/Users/dhirensarin/Dev/project-ada/features/configs/knn_technical.json"){
  json_data <- fromJSON(file=path)
  class(json_data) <- "FeatureReader"
  return (json_data)
}

#' Generic Function dispatch
getFrequency <- function (x, ...) {
  UseMethod("getFrequency", x)
}

#' Generic Function dispatch
getColumns<- function (x, ...) {
  UseMethod("getColumns", x)
}

#'Fetch specific ticker history from flat file
#'@return zoo
getFrequency.FeatureReader <- function(x) {
  return (x$frequency)
}

#'Fetch specific ticker history from flat file
#'@return zoo
getColumns.FeatureReader <- function(x) {
  return (x$feature_columns)
}

