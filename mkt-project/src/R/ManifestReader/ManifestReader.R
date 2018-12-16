#'Read manifest files from a directory to identify data sources
#'@param path to manifest files
#'@return DataReader object
#'@examples
#'dr <- ManifestReader()
ManifestReader <- function(path = "/Users/dhirensarin/Dev/project-ada/data/configs"){
  files <- list.files(path)
  json_data <- list()
  for (i in files) {
    print(i)
    json_data[[i]] <- fromJSON(file=paste0(path, "/", i))
  }
  class(json_data) <- "ManifestReader"
  return (json_data)
}

#' Generic Function dispatch to get appropriate data source by Id
getDataReader <- function (x, ...) {
  UseMethod("getDataReader", x)
}

#' Generic Function dispatch to get appropriate data source by Id
getFeatureReader <- function (x, ...) {
  UseMethod("getFeatureReader", x)
}

#'Fetch DataReader object given a raw_data_id
getDataReader.ManifestReader <- function(x, dataset_id="quandl:wiki/eod") {
  for (i in x) {
    if (i$dataset_id == dataset_id) {
      class(i) <- "DataReader"
      return (i)
    }
  }
}

#'Fetch vector of feature names for a given algorithm id
getFeatureReader.ManifestReader <- function(x, features_id="knn_1") {
  for (i in x) {
    if (i$features_id == features_id) {
      class(i) <- "FeatureReader"
      return (i)
    }
  }
}
