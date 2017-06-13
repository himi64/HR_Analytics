# File: HRanalytics.R
# User: Himanshu Makharia
# HRdata obtained from Kaggle

# This R file contains various classification methods used to predict whether
# an employee left the company based on various parameters. 


################################################
### DATA PREPROCESSING 

# read data
HRdata = read.csv("HRdata.csv")

# check each column for NA entries
sum(is.na(HRdata$satisfaction_level))
sum(is.na(HRdata$last_evaluation))
sum(is.na(HRdata$number_project))
sum(is.na(HRdata$average_montly_hours))
sum(is.na(HRdata$time_spend_company))
sum(is.na(HRdata$Work_accident))
sum(is.na(HRdata$promotion_last_5years))
sum(is.na(HRdata$sales))
sum(is.na(HRdata$salary))
sum(is.na(HRdata$left))

HRdata = na.omit(HRdata)

# encode categorical variables
HRdata$sales = factor(HRdata$sales,
                       levels = c('hr', 'IT', 'management', 'marketing', 'product_mng', 'RanD', 'sales', 'support', 'technical', 'accounting'),
                       labels = c(1,2,3,4,5,6,7,8,9,10))

HRdata$salary= factor(HRdata$salary,
                       levels = c('low', 'medium', 'high'),
                       labels = c(1,2,3))

# encode target feature as a factor (needed for naive bayes, but can be specified for other classifiers also)
HRdata$left = factor(HRdata$left,
                     levels = c('0','1'),
                     labels = c(0,1))

# split data into train and test
library(caTools)

split = sample.split(HRdata$left, SplitRatio = 0.8)
training_set = subset(HRdata, split == TRUE)
test_set = subset(HRdata, split == FALSE)

# scale the data (except the dependent variable) - only needed for data visualization
#training_set[1:7] = scale(training_set[1:7])
#test_set[1:7] = scale(test_set[1:7])

#################################################
### LOGISTIC REGRESSION 

# perform logistic regression on training data
logistic_reg = glm(left~., family=binomial, data=training_set) #excluding col 10 (dependent var)
summary(logistic_reg)

# predict test set results
logistic_reg_pred = predict(logistic_reg, type = 'response', newdata=test_set[-10])
logistic_reg_y_pred = ifelse(prediction > 0.5, 1, 0)

# create confusion matrix for accuracy test
logistic_reg_cm = table(test_set[, 10], logistic_reg_y_pred > 0.5)
logistic_reg_cm

# accuracy
(1996+252)/(1996+164+437+252)
# = 78% accuracy

#############################################
### DECISION TREE CLASSIFICATION

# build the decision tree classifier
install.packages(rpart) # install rpart package to plot trees
decisiontree = rpart(left~., data=training_set)
rpart.plot(decisiontree) #NB: remove scaling when plotting tree

# predict using the test set
decisiontree_pred = predict(decisiontree, newdata=test_set[-10])
decisiontree_y_pred = ifelse(prediction > 0.5, 1, 0) 
# assume 0.5 as the threshold for left/not left

# assess accuracy using confusion matrix
decisiontree_cm = table(test_set[, 10], decisiontree_y_pred)
decisiontree_cm

# accuracy
(2010+253)/(2010+160+434+253)
# = 79.2%'

#############################################
### RANDOM FOREST CLASSIFICATION (a type of enseble learning)

# encode the target variable as a factor (needs to be done for naive bayes also)
HRdata$left = factor(HRdata$left, levels = c(0,1))

# build the random forest classifier
library(randomForest)
?randomForest()
randomforest = randomForest(x=training_set[-10][-8],
                            y=training_set$left,
                            ntree=20) # need to exclude sales col because of NA

# predict test set results
randomforest_pred = predict(randomforest, newdata=test_set[-10][-8])
randomforest

# confusion matrix
randomforest_cm = table(test_set[, 10], randomforest_pred)
randomforest_cm #need to fix the CM

# accuracy
(2281+682)/(2281+682+52+5) # 98.11% for 100 trees
(2281+688)/(2281+688+26+5) # 98.97% for 200 trees

#############################################
### K-NEAREST NEIGHBOR CLASSIFICATION
library(class)

# build the knn classifier
knn_y_pred = knn(train = training_set[, -10],
                 test = test_set[, -10],
                 cl = training_set[, 10],
                 k = 5)

knn_y_pred

# confusion matrix
knn_cm = table(test_set[, 10], knn_y_pred)
knn_cm

# accuracy
(1995+623)/(1995+623+67+157)
# = 92.11% for k=5

#############################################
### NAIVE BAYES CLASSIFICATION

library(e1071)

# fit the naive bayes classifier to the training set
nb_classifier = naiveBayes(x = training_set[-10],
                           y = training_set$left)

# predict the test set results using algorithm
nb_predict = predict(nb_classifier, newdata = test_set[-10])
nb_predict 
# Naive bayes doesnt recognize the output (left) as a factor (binary)
# so you need to encode $left data as a factor in the data preprocessing step

# create confusion matrix
nb_cm = table(test_set[, 10], nb_predict)
nb_cm

# accuracy


#############################################
### SUPPORT VECTOR MACHINE CLASSIFICATION (LINEAR)
library(e1071)

# 1. build classifier using linear kernel
linear_svm = svm(left~., data=training_set, 
                 type = 'C-classification',
                 kernel = 'linear')

# predict on the test set
linear_svm_pred = predict(linear_svm, newdata = test_set[-10])
linear_svm_pred

# confusion matrix
linear_svm_cm = table(test_set[, 10], linear_svm_pred)
linear_svm_cm

# 2. build classifier using polynomial kernel
poly_svm = svm(left~., data=training_set, 
                 type = 'C-classification',
                 kernel = 'polynomial')

# predict on the test set
poly_svm_pred = predict(poly_svm, newdata = test_set)
poly_svm_pred

# confusion matrix
poly_svm_cm = table(test_set[, 10], poly_svm_pred)
poly_svm_cm

# 3. build classifier using radial basis kernel
radial_svm = svm(left~., data=training_set, 
                 type = 'C-classification',
                 kernel = 'radial')

# predict on the test set
radial_svm_pred = predict(radial_svm, newdata = test_set)
radial_svm_pred

# confusion matrix
radial_svm_cm = table(test_set[, 10], radial_svm_pred)
radial_svm_cm

# 4. build classifier using sigmoid kernel
sigmoid_svm = svm(left~., data=training_set, 
                 type = 'C-classification',
                 kernel = 'sigmoid')

# predict on the test set
sigmoid_svm_pred = predict(sigmoid_svm, newdata = test_set)
sigmoid_svm_pred

# confusion matrix
sigmoid_svm_cm = table(test_set[, 10], sigmoid_svm_pred)
sigmoid_svm_cm

# accuracy of various kernels:
# linear: 77.09%
# polynomial: 94.72%
# radial basis: 95.57%
# sigmoid: 57.74%

