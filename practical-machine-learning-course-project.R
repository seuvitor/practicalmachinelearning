library(data.table)
library(caret)
library(randomForest)
library(e1071)
library(parallel)
library(doParallel)

set.seed(15243)

harTrainingDataFileName <- "pml-training.csv"
harTrainingDataUrl <-
    "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

if (!file.exists(harTrainingDataFileName)) {
    download.file(harTrainingDataUrl, harTrainingDataFileName)
}

harData <- fread(harTrainingDataFileName, stringsAsFactors = FALSE)

# Function to remove columns that are not very useful for modeling
prepareData <- function (df) {
    percentNaInCols <- sapply(1:ncol(harData), function (i) {
        sum(is.na(harData[[i]])) / nrow(harData)
    })
    nonNaCols <- percentNaInCols < 0.9
    nonSensorColNames <- c("V1", "user_name", "raw_timestamp_part_1",
                           "raw_timestamp_part_2", "cvtd_timestamp",
                           "new_window", "num_window")

    df <- df[, nonNaCols, with=FALSE]
    df <- df[, -nonSensorColNames, with = FALSE]
    df
}

# Split data into training and test sets
inTrain <- createDataPartition(y=harData$classe, p=0.75, list=FALSE)

training <- harData[inTrain,]
testing <- harData[-inTrain,]

cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

numModelsEvaluated = 5
numFolds = 5

seeds <- vector(mode = "list", length = numFolds + 1)
for(i in 1:numFolds) seeds[[i]] <- sample.int(n = 1000, numModelsEvaluated)
seeds[[numFolds + 1]] <- sample.int(1000, 1)

modFit <- train(classe ~ .,
                data = prepareData(training),
                preProcess = c("center","scale"),
                method="rf",
                trControl = trainControl(method = "cv",
                                         number = numFolds,
                                         allowParallel = TRUE),
                tuneGrid = data.frame(mtry = numModelsEvaluated))

stopCluster(cluster)
registerDoSEQ()

print(modFit)

predictionOnTrainingSet <- predict(modFit, prepareData(training))
trainingSetAccuracy <- sum(predictionOnTrainingSet == training$classe) / nrow(training)
table(predictionOnTrainingSet, training$classe)

predictionOnTestSet <- predict(modFit, prepareData(testing))
testSetAccuracy <- sum(predictionOnTestSet == testing$classe) / nrow(testing)

table(predictionOnTestSet, testing$classe)
