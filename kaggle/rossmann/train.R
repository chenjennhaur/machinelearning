library(readr)
require(Matrix)
require(xgboost)


train <- read_csv('X_train.csv')
ytrain <- read_csv('y_train.csv')
test <- read_csv('X_test.csv')
id <- read_csv('labels.csv')

# train.full.sparse <- sparse.model.matrix(data=train)
  
rmpse <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))
  epreds <- exp(as.numeric(preds))
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

param <- list(
  objective="reg:linear",
  booster="gbtree",
  eta=0.02, # Control the learning rate
  max.depth=10, # Maximum depth of the tree
  subsample=0.9, # subsample ratio of the training instance
  colsample_bytree=0.7 # subsample ratio of columns when constructing each tree
)


# dtrain <- xgb.DMatrix(
  # data=train.full.sparse, 
  # label=ytrain)

# history <- xgb.cv(
  # data=dtrain,
  # params = param,
  # early.stop.round=30, # training with a validation set will stop if the performance keeps getting worse consecutively for k rounds
  # nthread=4, # number of CPU threads
  # nround=50, # number of trees
  # verbose=0, # do not show partial info
  # nfold=5, # number of CV folds
  # feval=rmpse, # custom evaluation metric
  # maximize=FALSE # the lower the evaluation score the better
# )

train[] <- lapply(train,as.numeric)
ytrain[] <- lapply(ytrain,as.numeric)
test[] <- lapply(test,as.numeric)
dtrain<-xgb.DMatrix(data=data.matrix(train),label=data.matrix(ytrain))

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 300, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                    maximize            = FALSE,
                    feval=rmpse
)
pred1 <- predict(clf, data.matrix(test))
submission <- data.frame(Id=id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, "rf1.csv")