---
title: "Analysis of Weight Lifting Exercise Quality"
author: "Vitor Dantas"
date: "7/13/2019"
output: 
  html_document: 
    keep_md: true 
---

## Synopsis

This report aims to explain the development of a model for predicting the
quality of Weight Lifting Exercise, based on data from sensors placed on the
body of the person. The data was made available by the
[LES group at PUC-Rio, Brazil](http://groupware.les.inf.puc-rio.br/har).

We did some preprocessing and developed a random forest model using the caret R
package, and according to our cross validation strategy, we expect the out of
sample error to be below 0.5%.

### Exploratory analysis and preprocessing





We load the data and notice that there are 19622 observations of
160 variables. While exploring, we see that many variables contain
mostly NA values, and the few first variables (V1, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window) are not
supposed to be used for predictions, since they are only meaningful in relation
to the specific experiment when the were recorded. So we make a helper function
`prepareData` to bypass those columns, leaving only the sensor-related variables
and the output variable `classe`. We will use this function before passing data
to build the model and also later when predicting.



For the random forest method, it wasn't really necessary to normalize the data,
but we use the parameter in train function `preProcess = c("center","scale")`
to achieve this for all features.

### Experimental setup


```r
# Split data into training and test sets
inTrain <- createDataPartition(y=harData$classe, p=0.75, list=FALSE)

training <- harData[inTrain,]
testing <- harData[-inTrain,]
```

We then split the data (which comes from the "training" file) into training and
test sets using the createDataPartition function with a 0.75 split. The testing
set will be kept aside during modelling and will be used at the end to evaluate
out of sample error.

### Training the Random Forest model

We have chosen the Random Forest method for its general accuracy and easy of
implementation, while sacrificing the interpretability of the model. Since the
inicial results were good, we kept our choice.

We used the `train` caret function to build our model, and after some
exploration we chose parameters to use a 5-folds cross validation, tuned the
random forest method with an mtry of 5, and we prepared the setup to run the
model in parallel for maximized speed (we also take care of using the seeds
parameter to allow for reproducibility). After training, we obtained the
following model:


```r
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
                                         allowParallel = TRUE,
                                         seeds),
                tuneGrid = data.frame(mtry = numModelsEvaluated))

stopCluster(cluster)
registerDoSEQ()

modFit
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11775, 11773, 11775, 11776, 11773 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.9922542  0.9902014
## 
## Tuning parameter 'mtry' was held constant at a value of 5
```

### Evaluation of the model

We then used our model to predict the values of the `classe` variable on the
training set and also on the kept aside test set.


```r
predictionOnTrainingSet <- predict(modFit, prepareData(training))
trainingSetAccuracy <- sum(predictionOnTrainingSet == training$classe) / nrow(training)

predictionOnTestSet <- predict(modFit, prepareData(testing))
testSetAccuracy <- sum(predictionOnTestSet == testing$classe) / nrow(testing)
```

As it was expected, the accuracy of the prediction over the the training set
(the resubstitution accuracy) was high, achieving
100%. More importantly, the accuracy over the test set
that was not used in modeling, 99.6737357% was also good enough
for our purposes, so we expect the accuracy for predicting over new data to be
similar. The following table shows the predictions that were missed on our test
set.


---------------------------------------
 &nbsp;    A      B     C     D     E  
-------- ------ ----- ----- ----- -----
 **A**    1394    5     0     0     0  

 **B**     1     943    3     0     0  

 **C**     0      1    852    5     0  

 **D**     0      0     0    798    0  

 **E**     0      0     0     1    901 
---------------------------------------
