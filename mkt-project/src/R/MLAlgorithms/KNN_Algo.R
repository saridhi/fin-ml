library(caret)
library(e1071)

#'KNN algorithm to sequentially determine predictions in a time_series as a classification problem
#'Training is performed on a rolling basis with feature data scaled
#'@param data_set matrix
#'@param retrain_gap retrain every x periods
#'@return time_series of predictions
doKnn <- function(data_set, retrain_gap=100) {
  preProcessParams <- preProcess(as.data.frame(data_set[,1:(NCOL(data_set)-1)]), method=c('scale'))
  trained_model <- NULL
  predictions <- rollapplyr(data_set, NROW(data_set), by.column=FALSE, partial=TRUE, by=retrain_gap,
                            function(x, preProcessParams, retrain_gap) {
                              if (NROW(x)<20) { #Threshold for minimum samples needed
                                #trained_model <<- NULL
                                return (0)
                              } else {
                                #if (is.integer(NROW(x)/retrain_gap) | is.null(trained_model)) {
                                trained_model <- knn_train_model(x[1:NROW(x)-1,], preProcessParams)
                                #} 
                                prediction <- predict(object=trained_model, 
                                                      as.data.frame(x[(NROW(x)-retrain_gap):NROW(x),(1:(NCOL(x)-1))]))
                                print(NROW(x))
                                if (retrain_gap==1) {
                                  prediction <- prediction[2]
                                } else {
                                  prediction
                                }
                                as.numeric(as.character(prediction))
                              }
                            }, preProcessParams = preProcessParams, retrain_gap=retrain_gap)
  zoo(as.vector(predictions), order.by=index(data_set))
}


#'Function to train knn model
#'@param data_subset applied on a rolling basis
#'@param preProcessParams any instructions to scale/center data etc
#'@return trained model
knn_train_model <- function(data_subset, preProcessParams = NULL) {
  train_series <- predict(preProcessParams, data_subset[,1:(NCOL(data_subset)-1)])
  trained_model <- train(data_subset[,1:(NCOL(data_subset)-1)],
                         as.factor(data_subset[,NCOL(data_subset)]),
                         method='knn')
}
