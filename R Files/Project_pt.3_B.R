#######################################################
#Statistical Learning Project - Pt.3 with balancing techniques
#Capelletti Thomas - 726582
#######################################################


#######################################################
#Delete all from the Environment and load the libraries
#######################################################
rm(list=ls())

library(class)
library(dplyr)
library(ggplot2)
library(gridExtra) 
library(corrplot)
library(ppcor)
library(car)
library(ISLR2)
library(ROCR)
library(MASS)
library(caret)
library(naivebayes)
library(e1071)
library(pROC)


#######################################################
#######################################################
#MODEL NUMBER 1 - SMOTE#
#######################################################
#######################################################

#######################################################
#Dataset Loading
#######################################################
getwd()
setwd("C:\\Users\\thoma\\OneDrive - unibs.it\\Statistical Learning\\Project\\Project_CT_726582")
dataset <- read.csv("loan_data_formatted3S.csv")

#Defining Training and Test Sets:
n <- nrow(dataset)
set.seed(0) #Set seed to ensure reproducibility 
train.ind <- sample(1:n, size = 0.75*n)

#The Training set will consist of 75% of the records and will be used for data analysis and exploration.
training_set <- dataset[train.ind,]
nrow(training_set) #369 observations

#The Test set will consist of the remaining 25% and will be used to test the various 
#models addressed during the course with respective evaluation of the results.
test_set <- dataset[-train.ind,]
nrow(test_set) #124 observations

#Count
count_d <- dataset %>%
  count(Loan_Status)
print(count_d) #Rejected = 260 and Accepted = 234

#Logistic Regression: Suitable for binary classification problems with linear relationships between predictors and the log-odds of the response.
#LDA: Suitable for classification with multivariate normal distributions and same covariance matrices for all classes.
#QDA: Suitable for classification with multivariate normal distributions and different covariance matrices for each class.
#Naive Bayes: Suitable for classification with the assumption of independence between features, simple and efficient, but may not be accurate if features are correlated.

#######################################################
#Logistic Regression M1
#######################################################
logistic_m1 <- glm(Loan_Status ~ Credit_History, data = training_set, family = binomial)
summary(logistic_m1)
log_m1.test <- predict(logistic_m1, test_set, type="response")
log_m1.pred <- rep("No", nrow(test_set))
log_m1.pred[log_m1.test > 0.60] = "Yes"
log_nvector_m1 <- as.numeric(log_m1.pred == "Yes")
log_fvector_m1 <- factor(log_nvector_m1, levels = c(0, 1))
print(log_fvector_m1)
confusionMatrix(data=as.factor(log_nvector_m1),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m1 <- prediction(log_m1.test,as.factor(test_set$Loan_Status))
perflg_m1 <- performance(predlg_m1,"tpr","fpr")
plot(perflg_m1, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m1_auc<- performance(predlg_m1, "auc")@y.values[[1]]
log_m1_auc

#######################################################
#LDA M1
#######################################################
lda_m1 <- lda(Loan_Status ~ Credit_History, data = training_set)
lda_m1.pred <- predict(lda_m1, newdata = test_set)
lda_m1.prob <- lda_m1.pred$posterior[, "1"]
lda_m1.trs = 0.70
lda_m1.pred_class<- ifelse(lda_m1.prob > lda_m1.trs, "Yes", "No")
lda_m1.nvector <- as.numeric(lda_m1.pred_class == "Yes")
lda_m1.fvector <- factor(lda_m1.nvector, levels = c(0, 1))
print(lda_m1.fvector)
confusionMatrix(data = lda_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m1 <- prediction(lda_m1.prob,as.factor(test_set$Loan_Status))
perflda_m1 <- performance(predlda_m1,"tpr","fpr")
plot(perflda_m1, main="ROC curve for LDA", colorize=TRUE)
lda_m1_auc <- performance(predlda_m1, "auc")@y.values[[1]]
lda_m1_auc

#######################################################
#QDA M1
#######################################################
qda_m1 <- qda(Loan_Status ~ Credit_History, data = training_set)
qda_m1.pred <- predict(qda_m1, newdata = test_set)
qda_m1.prob <- qda_m1.pred$posterior[, "1"]
qda_m1.trs = 0.60
qda_m1.pred_class <- ifelse(qda_m1.prob > qda_m1.trs, "Yes", "No")
qda_m1.nvector <- as.numeric(qda_m1.pred_class == "Yes")
qda_m1.fvector <- factor(qda_m1.nvector, levels = c(0, 1))
print(qda_m1.fvector)
confusionMatrix(data = qda_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m1 <- prediction(qda_m1.prob,as.factor(test_set$Loan_Status))
perfqda_m1 <- performance(predqda_m1,"tpr","fpr")
plot(perfqda_m1, main="ROC curve for QDA", colorize=TRUE)
qda_m1_auc <- performance(predqda_m1, "auc")@y.values[[1]]
qda_m1_auc

#######################################################
#Naive Bayes M1
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m1 <- naiveBayes(Loan_Status ~ Credit_History, data = training_set)
nb_m1.pred <- predict(nb_m1, newdata = test_set, type = "raw")
nb_m1.prob <- nb_m1.pred[, 2]
nb_m1.trs = 0.60
nb_m1.pred_class <- ifelse(nb_m1.prob > nb_m1.trs, "Yes", "No")
nb_m1.nvector <- as.numeric(nb_m1.pred_class == "Yes")
nb_m1.fvector <- factor(nb_m1.nvector, levels = c(0, 1))
print(nb_m1.fvector)
confusionMatrix(data = nb_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m1 <- prediction(nb_m1.prob,as.factor(test_set$Loan_Status))
perfnb_m1 <- performance(prednb_m1,"tpr","fpr")
plot(perfnb_m1, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m1_auc <- performance(prednb_m1, "auc")@y.values[[1]]
nb_m1_auc

#######################################################
#Plot M1
#######################################################
plot(perflg_m1, main = "ROC curves M1", col = "blue", lwd = 2)
lines(perflda_m1@x.values[[1]], perflda_m1@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m1@x.values[[1]], perfqda_m1@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m1@x.values[[1]], perfnb_m1@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m1_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m1_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m1_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m1_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M2
#######################################################
logistic_m2 <- glm(Loan_Status ~ Credit_History + Semiurban, data = training_set, family = binomial)
summary(logistic_m2)
log_m2.test <- predict(logistic_m2, test_set, type="response")
log_m2.pred <- rep("No", nrow(test_set))
log_m2.pred[log_m2.test > 0.49] = "Yes"
log_nvector_m2 <- as.numeric(log_m2.pred == "Yes")
log_fvector_m2 <- factor(log_nvector_m2, levels = c(0, 1))
print(log_fvector_m2)
confusionMatrix(data=as.factor(log_nvector_m2),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m2 <- prediction(log_m2.test,as.factor(test_set$Loan_Status))
perflg_m2 <- performance(predlg_m2,"tpr","fpr")
plot(perflg_m2, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m2_auc<- performance(predlg_m2, "auc")@y.values[[1]]
log_m2_auc

#######################################################
#LDA M2
#######################################################
lda_m2 <- lda(Loan_Status ~ Credit_History + Semiurban, data = training_set)
lda_m2.pred <- predict(lda_m2, newdata = test_set)
lda_m2.prob <- lda_m2.pred$posterior[, "1"]
lda_m2.trs = 0.65
lda_m2.pred_class<- ifelse(lda_m2.prob > lda_m2.trs, "Yes", "No")
lda_m2.nvector <- as.numeric(lda_m2.pred_class == "Yes")
lda_m2.fvector <- factor(lda_m2.nvector, levels = c(0, 1))
print(lda_m2.fvector)
confusionMatrix(data = lda_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m2 <- prediction(lda_m2.prob,as.factor(test_set$Loan_Status))
perflda_m2 <- performance(predlda_m2,"tpr","fpr")
plot(perflda_m2, main="ROC curve for LDA", colorize=TRUE)
lda_m2_auc <- performance(predlda_m2, "auc")@y.values[[1]]
lda_m2_auc

#######################################################
#QDA M2
#######################################################
qda_m2 <- qda(Loan_Status ~ Credit_History + Semiurban, data = training_set)
qda_m2.pred <- predict(qda_m2, newdata = test_set)
qda_m2.prob <- qda_m2.pred$posterior[, "1"]
qda_m2.trs = 0.65
qda_m2.pred_class <- ifelse(qda_m2.prob > qda_m2.trs, "Yes", "No")
qda_m2.nvector <- as.numeric(qda_m2.pred_class == "Yes")
qda_m2.fvector <- factor(qda_m2.nvector, levels = c(0, 1))
print(qda_m2.fvector)
confusionMatrix(data = qda_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m2 <- prediction(qda_m2.prob,as.factor(test_set$Loan_Status))
perfqda_m2 <- performance(predqda_m2,"tpr","fpr")
plot(perfqda_m2, main="ROC curve for QDA", colorize=TRUE)
qda_m2_auc <- performance(predqda_m2, "auc")@y.values[[1]]
qda_m2_auc

#######################################################
#Naive Bayes M2
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m2 <- naiveBayes(Loan_Status ~ Credit_History + Semiurban, data = training_set)
nb_m2.pred <- predict(nb_m2, newdata = test_set, type = "raw")
nb_m2.prob <- nb_m2.pred[, 2]
nb_m2.trs = 0.63
nb_m2.pred_class <- ifelse(nb_m2.prob > nb_m2.trs, "Yes", "No")
nb_m2.nvector <- as.numeric(nb_m2.pred_class == "Yes")
nb_m2.fvector <- factor(nb_m2.nvector, levels = c(0, 1))
print(nb_m2.fvector)
confusionMatrix(data = nb_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m2 <- prediction(nb_m2.prob,as.factor(test_set$Loan_Status))
perfnb_m2 <- performance(prednb_m2,"tpr","fpr")
plot(perfnb_m2, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m2_auc <- performance(prednb_m2, "auc")@y.values[[1]]
nb_m2_auc

#######################################################
#Plot M2
#######################################################
plot(perflg_m2, main = "ROC curves m2", col = "blue", lwd = 2)
lines(perflda_m2@x.values[[1]], perflda_m2@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m2@x.values[[1]], perfqda_m2@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m2@x.values[[1]], perfnb_m2@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m2_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m2_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m2_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m2_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M3
#######################################################
logistic_m3 <- glm(Loan_Status ~ Married + Credit_History + Semiurban, data = training_set, family = binomial)
summary(logistic_m3)
log_m3.test <- predict(logistic_m3, test_set, type="response")
log_m3.pred <- rep("No", nrow(test_set))
log_m3.pred[log_m3.test > 0.60] = "Yes"
log_nvector_m3 <- as.numeric(log_m3.pred == "Yes")
log_fvector_m3 <- factor(log_nvector_m3, levels = c(0, 1))
print(log_fvector_m3)
confusionMatrix(data=as.factor(log_nvector_m3),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m3 <- prediction(log_m3.test,as.factor(test_set$Loan_Status))
perflg_m3 <- performance(predlg_m3,"tpr","fpr")
plot(perflg_m3, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m3_auc<- performance(predlg_m3, "auc")@y.values[[1]]
log_m3_auc

#######################################################
#LDA M3
#######################################################
lda_m3 <- lda(Loan_Status ~ Married + Credit_History + Semiurban, data = training_set)
lda_m3.pred <- predict(lda_m3, newdata = test_set)
lda_m3.prob <- lda_m3.pred$posterior[, "1"]
lda_m3.trs = 0.63
lda_m3.pred_class<- ifelse(lda_m3.prob > lda_m3.trs, "Yes", "No")
lda_m3.nvector <- as.numeric(lda_m3.pred_class == "Yes")
lda_m3.fvector <- factor(lda_m3.nvector, levels = c(0, 1))
print(lda_m3.fvector)
confusionMatrix(data = lda_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m3 <- prediction(lda_m3.prob,as.factor(test_set$Loan_Status))
perflda_m3 <- performance(predlda_m3,"tpr","fpr")
plot(perflda_m3, main="ROC curve for LDA", colorize=TRUE)
lda_m3_auc <- performance(predlda_m3, "auc")@y.values[[1]]
lda_m3_auc

#######################################################
#QDA M3
#######################################################
qda_m3 <- qda(Loan_Status ~ Married + Credit_History + Semiurban, data = training_set)
qda_m3.pred <- predict(qda_m3, newdata = test_set)
qda_m3.prob <- qda_m3.pred$posterior[, "1"]
qda_m3.trs = 0.60
qda_m3.pred_class <- ifelse(qda_m3.prob > qda_m3.trs, "Yes", "No")
qda_m3.nvector <- as.numeric(qda_m3.pred_class == "Yes")
qda_m3.fvector <- factor(qda_m3.nvector, levels = c(0, 1))
print(qda_m3.fvector)
confusionMatrix(data = qda_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m3 <- prediction(qda_m3.prob,as.factor(test_set$Loan_Status))
perfqda_m3 <- performance(predqda_m3,"tpr","fpr")
plot(perfqda_m3, main="ROC curve for QDA", colorize=TRUE)
qda_m3_auc <- performance(predqda_m3, "auc")@y.values[[1]]
qda_m3_auc

#######################################################
#Naive Bayes M3
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m3 <- naiveBayes(Loan_Status ~ Married + Credit_History + Semiurban, data = training_set)
nb_m3.pred <- predict(nb_m3, newdata = test_set, type = "raw")
nb_m3.prob <- nb_m3.pred[, 2]
nb_m3.trs = 0.60
nb_m3.pred_class <- ifelse(nb_m3.prob > nb_m3.trs, "Yes", "No")
nb_m3.nvector <- as.numeric(nb_m3.pred_class == "Yes")
nb_m3.fvector <- factor(nb_m3.nvector, levels = c(0, 1))
print(nb_m3.fvector)
confusionMatrix(data = nb_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m3 <- prediction(nb_m3.prob,as.factor(test_set$Loan_Status))
perfnb_m3 <- performance(prednb_m3,"tpr","fpr")
plot(perfnb_m3, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m3_auc <- performance(prednb_m3, "auc")@y.values[[1]]
nb_m3_auc

#######################################################
#Plot M3
#######################################################
plot(perflg_m3, main = "ROC curves m3", col = "blue", lwd = 2)
lines(perflda_m3@x.values[[1]], perflda_m3@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m3@x.values[[1]], perfqda_m3@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m3@x.values[[1]], perfnb_m3@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m3_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m3_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m3_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m3_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M4
#######################################################
logistic_m4 <- glm(Loan_Status ~ Married + Credit_History + Semiurban + Rural, data = training_set, family = binomial)
summary(logistic_m4)
log_m4.test <- predict(logistic_m4, test_set, type="response")
log_m4.pred <- rep("No", nrow(test_set))
log_m4.pred[log_m4.test > 0.60] = "Yes"
log_nvector_m4 <- as.numeric(log_m4.pred == "Yes")
log_fvector_m4 <- factor(log_nvector_m4, levels = c(0, 1))
print(log_fvector_m4)
confusionMatrix(data=as.factor(log_nvector_m4),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m4 <- prediction(log_m4.test,as.factor(test_set$Loan_Status))
perflg_m4 <- performance(predlg_m4,"tpr","fpr")
plot(perflg_m4, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m4_auc<- performance(predlg_m4, "auc")@y.values[[1]]
log_m4_auc

#######################################################
#LDA M4
#######################################################
lda_m4 <- lda(Loan_Status ~ Married + Credit_History + Semiurban + Rural, data = training_set)
lda_m4.pred <- predict(lda_m4, newdata = test_set)
lda_m4.prob <- lda_m4.pred$posterior[, "1"]
lda_m4.trs = 0.60
lda_m4.pred_class<- ifelse(lda_m4.prob > lda_m4.trs, "Yes", "No")
lda_m4.nvector <- as.numeric(lda_m4.pred_class == "Yes")
lda_m4.fvector <- factor(lda_m4.nvector, levels = c(0, 1))
print(lda_m4.fvector)
confusionMatrix(data = lda_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m4 <- prediction(lda_m4.prob,as.factor(test_set$Loan_Status))
perflda_m4 <- performance(predlda_m4,"tpr","fpr")
plot(perflda_m4, main="ROC curve for LDA", colorize=TRUE)
lda_m4_auc <- performance(predlda_m4, "auc")@y.values[[1]]
lda_m4_auc

#######################################################
#QDA M4
#######################################################
qda_m4 <- qda(Loan_Status ~ Married + Credit_History + Semiurban + Rural, data = training_set)
qda_m4.pred <- predict(qda_m4, newdata = test_set)
qda_m4.prob <- qda_m4.pred$posterior[, "1"]
qda_m4.trs = 0.60
qda_m4.pred_class <- ifelse(qda_m4.prob > qda_m4.trs, "Yes", "No")
qda_m4.nvector <- as.numeric(qda_m4.pred_class == "Yes")
qda_m4.fvector <- factor(qda_m4.nvector, levels = c(0, 1))
print(qda_m4.fvector)
confusionMatrix(data = qda_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m4 <- prediction(qda_m4.prob,as.factor(test_set$Loan_Status))
perfqda_m4 <- performance(predqda_m4,"tpr","fpr")
plot(perfqda_m4, main="ROC curve for QDA", colorize=TRUE)
qda_m4_auc <- performance(predqda_m4, "auc")@y.values[[1]]
qda_m4_auc

#######################################################
#Naive Bayes M4
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m4 <- naiveBayes(Loan_Status ~ Married + Credit_History + Semiurban + Rural, data = training_set)
nb_m4.pred <- predict(nb_m4, newdata = test_set, type = "raw")
nb_m4.prob <- nb_m4.pred[, 2]
nb_m4.trs = 0.60
nb_m4.pred_class <- ifelse(nb_m4.prob > nb_m4.trs, "Yes", "No")
nb_m4.nvector <- as.numeric(nb_m4.pred_class == "Yes")
nb_m4.fvector <- factor(nb_m4.nvector, levels = c(0, 1))
print(nb_m4.fvector)
confusionMatrix(data = nb_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m4 <- prediction(nb_m4.prob,as.factor(test_set$Loan_Status))
perfnb_m4 <- performance(prednb_m4,"tpr","fpr")
plot(perfnb_m4, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m4_auc <- performance(prednb_m4, "auc")@y.values[[1]]
nb_m4_auc

#######################################################
#Plot M4
#######################################################
plot(perflg_m4, main = "ROC curves m4", col = "blue", lwd = 2)
lines(perflda_m4@x.values[[1]], perflda_m4@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m4@x.values[[1]], perfqda_m4@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m4@x.values[[1]], perfnb_m4@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m4_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m4_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m4_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m4_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M5
#######################################################
logistic_m5 <- glm(Loan_Status ~ Married + Education + Credit_History + Semiurban + Rural, data = training_set, family = binomial)
summary(logistic_m5)
log_m5.test <- predict(logistic_m5, test_set, type="response")
log_m5.pred <- rep("No", nrow(test_set))
log_m5.pred[log_m5.test > 0.60] = "Yes"
log_nvector_m5 <- as.numeric(log_m5.pred == "Yes")
log_fvector_m5 <- factor(log_nvector_m5, levels = c(0, 1))
print(log_fvector_m5)
confusionMatrix(data=as.factor(log_nvector_m5),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m5 <- prediction(log_m5.test,as.factor(test_set$Loan_Status))
perflg_m5 <- performance(predlg_m5,"tpr","fpr")
plot(perflg_m5, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m5_auc<- performance(predlg_m5, "auc")@y.values[[1]]
log_m5_auc

#######################################################
#LDA M5
#######################################################
lda_m5 <- lda(Loan_Status ~ Married + Education + Credit_History + Semiurban + Rural, data = training_set)
lda_m5.pred <- predict(lda_m5, newdata = test_set)
lda_m5.prob <- lda_m5.pred$posterior[, "1"]
lda_m5.trs = 0.60
lda_m5.pred_class<- ifelse(lda_m5.prob > lda_m5.trs, "Yes", "No")
lda_m5.nvector <- as.numeric(lda_m5.pred_class == "Yes")
lda_m5.fvector <- factor(lda_m5.nvector, levels = c(0, 1))
print(lda_m5.fvector)
confusionMatrix(data = lda_m5.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m5 <- prediction(lda_m5.prob,as.factor(test_set$Loan_Status))
perflda_m5 <- performance(predlda_m5,"tpr","fpr")
plot(perflda_m5, main="ROC curve for LDA", colorize=TRUE)
lda_m5_auc <- performance(predlda_m5, "auc")@y.values[[1]]
lda_m5_auc

#######################################################
#QDA M5
#######################################################
qda_m5 <- qda(Loan_Status ~ Married + Education + Credit_History + Semiurban + Rural, data = training_set)
qda_m5.pred <- predict(qda_m5, newdata = test_set)
qda_m5.prob <- qda_m5.pred$posterior[, "1"]
qda_m5.trs = 0.60
qda_m5.pred_class <- ifelse(qda_m5.prob > qda_m5.trs, "Yes", "No")
qda_m5.nvector <- as.numeric(qda_m5.pred_class == "Yes")
qda_m5.fvector <- factor(qda_m5.nvector, levels = c(0, 1))
print(qda_m5.fvector)
confusionMatrix(data = qda_m5.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m5 <- prediction(qda_m5.prob,as.factor(test_set$Loan_Status))
perfqda_m5 <- performance(predqda_m5,"tpr","fpr")
plot(perfqda_m5, main="ROC curve for QDA", colorize=TRUE)
qda_m5_auc <- performance(predqda_m5, "auc")@y.values[[1]]
qda_m5_auc

#######################################################
#Naive Bayes M5
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m5 <- naiveBayes(Loan_Status ~ Married + Education + Credit_History + Semiurban + Rural, data = training_set)
nb_m5.pred <- predict(nb_m5, newdata = test_set, type = "raw")
nb_m5.prob <- nb_m5.pred[, 2]
nb_m5.trs = 0.60
nb_m5.pred_class <- ifelse(nb_m5.prob > nb_m5.trs, "Yes", "No")
nb_m5.nvector <- as.numeric(nb_m5.pred_class == "Yes")
nb_m5.fvector <- factor(nb_m5.nvector, levels = c(0, 1))
print(nb_m5.fvector)
confusionMatrix(data = nb_m5.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m5 <- prediction(nb_m5.prob,as.factor(test_set$Loan_Status))
perfnb_m5 <- performance(prednb_m5,"tpr","fpr")
plot(perfnb_m5, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m5_auc <- performance(prednb_m5, "auc")@y.values[[1]]
nb_m5_auc

#######################################################
#Plot M5
#######################################################
plot(perflg_m5, main = "ROC curves m5", col = "blue", lwd = 2)
lines(perflda_m5@x.values[[1]], perflda_m5@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m5@x.values[[1]], perfqda_m5@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m5@x.values[[1]], perfnb_m5@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m5_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m5_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m5_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m5_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M6
#######################################################
logistic_m6 <- glm(Loan_Status ~ Applicant_Income + Married + Education + Credit_History + Semiurban + Rural, data = training_set, family = binomial)
summary(logistic_m6)
log_m6.test <- predict(logistic_m6, test_set, type="response")
log_m6.pred <- rep("No", nrow(test_set))
log_m6.pred[log_m6.test > 0.60] = "Yes"
log_nvector_m6 <- as.numeric(log_m6.pred == "Yes")
log_fvector_m6 <- factor(log_nvector_m6, levels = c(0, 1))
print(log_fvector_m6)
confusionMatrix(data=as.factor(log_nvector_m6),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m6 <- prediction(log_m6.test,as.factor(test_set$Loan_Status))
perflg_m6 <- performance(predlg_m6,"tpr","fpr")
plot(perflg_m6, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m6_auc<- performance(predlg_m6, "auc")@y.values[[1]]
log_m6_auc

#######################################################
#LDA M6
#######################################################
lda_m6 <- lda(Loan_Status ~ Applicant_Income + Married + Education + Credit_History + Semiurban + Rural, data = training_set)
lda_m6.pred <- predict(lda_m6, newdata = test_set)
lda_m6.prob <- lda_m6.pred$posterior[, "1"]
lda_m6.trs = 0.60
lda_m6.pred_class<- ifelse(lda_m6.prob > lda_m6.trs, "Yes", "No")
lda_m6.nvector <- as.numeric(lda_m6.pred_class == "Yes")
lda_m6.fvector <- factor(lda_m6.nvector, levels = c(0, 1))
print(lda_m6.fvector)
confusionMatrix(data = lda_m6.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m6 <- prediction(lda_m6.prob,as.factor(test_set$Loan_Status))
perflda_m6 <- performance(predlda_m6,"tpr","fpr")
plot(perflda_m6, main="ROC curve for LDA", colorize=TRUE)
lda_m6_auc <- performance(predlda_m6, "auc")@y.values[[1]]
lda_m6_auc

#######################################################
#QDA M6
#######################################################
qda_m6 <- qda(Loan_Status ~ Applicant_Income + Married + Education + Credit_History + Semiurban + Rural, data = training_set)
qda_m6.pred <- predict(qda_m6, newdata = test_set)
qda_m6.prob <- qda_m6.pred$posterior[, "1"]
qda_m6.trs = 0.60
qda_m6.pred_class <- ifelse(qda_m6.prob > qda_m6.trs, "Yes", "No")
qda_m6.nvector <- as.numeric(qda_m6.pred_class == "Yes")
qda_m6.fvector <- factor(qda_m6.nvector, levels = c(0, 1))
print(qda_m6.fvector)
confusionMatrix(data = qda_m6.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m6 <- prediction(qda_m6.prob,as.factor(test_set$Loan_Status))
perfqda_m6 <- performance(predqda_m6,"tpr","fpr")
plot(perfqda_m6, main="ROC curve for QDA", colorize=TRUE)
qda_m6_auc <- performance(predqda_m6, "auc")@y.values[[1]]
qda_m6_auc

#######################################################
#Naive Bayes M6
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m6 <- naiveBayes(Loan_Status ~ Applicant_Income + Married + Education + Credit_History + Semiurban + Rural, data = training_set)
nb_m6.pred <- predict(nb_m6, newdata = test_set, type = "raw")
nb_m6.prob <- nb_m6.pred[, 2]
nb_m6.trs = 0.60
nb_m6.pred_class <- ifelse(nb_m6.prob > nb_m6.trs, "Yes", "No")
nb_m6.nvector <- as.numeric(nb_m6.pred_class == "Yes")
nb_m6.fvector <- factor(nb_m6.nvector, levels = c(0, 1))
print(nb_m6.fvector)
confusionMatrix(data = nb_m6.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m6 <- prediction(nb_m6.prob,as.factor(test_set$Loan_Status))
perfnb_m6 <- performance(prednb_m6,"tpr","fpr")
plot(perfnb_m6, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m6_auc <- performance(prednb_m6, "auc")@y.values[[1]]
nb_m6_auc

#######################################################
#Plot M6
#######################################################
plot(perflg_m6, main = "ROC curves m6", col = "blue", lwd = 2)
lines(perflda_m6@x.values[[1]], perflda_m6@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m6@x.values[[1]], perfqda_m6@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m6@x.values[[1]], perfnb_m6@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m6_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m6_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m6_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m6_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M7
#######################################################
logistic_m7 <- glm(Loan_Status ~ Applicant_Income + Gender + Married + Education + Credit_History + Semiurban + Rural, data = training_set, family = binomial)
summary(logistic_m7)
log_m7.test <- predict(logistic_m7, test_set, type="response")
log_m7.pred <- rep("No", nrow(test_set))
log_m7.pred[log_m7.test > 0.70] = "Yes"
log_nvector_m7 <- as.numeric(log_m7.pred == "Yes")
log_fvector_m7 <- factor(log_nvector_m7, levels = c(0, 1))
print(log_fvector_m7)
confusionMatrix(data=as.factor(log_nvector_m7),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m7 <- prediction(log_m7.test,as.factor(test_set$Loan_Status))
perflg_m7 <- performance(predlg_m7,"tpr","fpr")
plot(perflg_m7, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m7_auc<- performance(predlg_m7, "auc")@y.values[[1]]
log_m7_auc

#######################################################
#LDA M7
#######################################################
lda_m7 <- lda(Loan_Status ~ Applicant_Income + Gender + Married + Education + Credit_History + Semiurban + Rural, data = training_set)
lda_m7.pred <- predict(lda_m7, newdata = test_set)
lda_m7.prob <- lda_m7.pred$posterior[, "1"]
lda_m7.trs = 0.70
lda_m7.pred_class<- ifelse(lda_m7.prob > lda_m7.trs, "Yes", "No")
lda_m7.nvector <- as.numeric(lda_m7.pred_class == "Yes")
lda_m7.fvector <- factor(lda_m7.nvector, levels = c(0, 1))
print(lda_m7.fvector)
confusionMatrix(data = lda_m7.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m7 <- prediction(lda_m7.prob,as.factor(test_set$Loan_Status))
perflda_m7 <- performance(predlda_m7,"tpr","fpr")
plot(perflda_m7, main="ROC curve for LDA", colorize=TRUE)
lda_m7_auc <- performance(predlda_m7, "auc")@y.values[[1]]
lda_m7_auc

#######################################################
#QDA M7
#######################################################
qda_m7 <- qda(Loan_Status ~ Applicant_Income + Gender + Married + Education + Credit_History + Semiurban + Rural, data = training_set)
qda_m7.pred <- predict(qda_m7, newdata = test_set)
qda_m7.prob <- qda_m7.pred$posterior[, "1"]
qda_m7.trs = 0.70
qda_m7.pred_class <- ifelse(qda_m7.prob > qda_m7.trs, "Yes", "No")
qda_m7.nvector <- as.numeric(qda_m7.pred_class == "Yes")
qda_m7.fvector <- factor(qda_m7.nvector, levels = c(0, 1))
print(qda_m7.fvector)
confusionMatrix(data = qda_m7.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m7 <- prediction(qda_m7.prob,as.factor(test_set$Loan_Status))
perfqda_m7 <- performance(predqda_m7,"tpr","fpr")
plot(perfqda_m7, main="ROC curve for QDA", colorize=TRUE)
qda_m7_auc <- performance(predqda_m7, "auc")@y.values[[1]]
qda_m7_auc

#######################################################
#Naive Bayes M7
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m7 <- naiveBayes(Loan_Status ~ Applicant_Income + Gender + Married + Education + Credit_History + Semiurban + Rural, data = training_set)
nb_m7.pred <- predict(nb_m7, newdata = test_set, type = "raw")
nb_m7.prob <- nb_m7.pred[, 2]
nb_m7.trs = 0.70
nb_m7.pred_class <- ifelse(nb_m7.prob > nb_m7.trs, "Yes", "No")
nb_m7.nvector <- as.numeric(nb_m7.pred_class == "Yes")
nb_m7.fvector <- factor(nb_m7.nvector, levels = c(0, 1))
print(nb_m7.fvector)
confusionMatrix(data = nb_m7.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m7 <- prediction(nb_m7.prob,as.factor(test_set$Loan_Status))
perfnb_m7 <- performance(prednb_m7,"tpr","fpr")
plot(perfnb_m7, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m7_auc <- performance(prednb_m7, "auc")@y.values[[1]]
nb_m7_auc

#######################################################
#Plot M7
#######################################################
plot(perflg_m7, main = "ROC curves m7", col = "blue", lwd = 2)
lines(perflda_m7@x.values[[1]], perflda_m7@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m7@x.values[[1]], perfqda_m7@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m7@x.values[[1]], perfnb_m7@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m7_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m7_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m7_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m7_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M8
#######################################################
logistic_m8 <- glm(Loan_Status ~ Applicant_Income + Gender + Married + Education + Credit_History + Semiurban + Rural + Family_I_D, data = training_set, family = binomial)
summary(logistic_m8)
log_m8.test <- predict(logistic_m8, test_set, type="response")
log_m8.pred <- rep("No", nrow(test_set))
log_m8.pred[log_m8.test > 0.80] = "Yes"
log_nvector_m8 <- as.numeric(log_m8.pred == "Yes")
log_fvector_m8 <- factor(log_nvector_m8, levels = c(0, 1))
print(log_fvector_m8)
confusionMatrix(data=as.factor(log_nvector_m8),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m8 <- prediction(log_m8.test,as.factor(test_set$Loan_Status))
perflg_m8 <- performance(predlg_m8,"tpr","fpr")
plot(perflg_m8, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m8_auc<- performance(predlg_m8, "auc")@y.values[[1]]
log_m8_auc

#######################################################
#LDA M8
#######################################################
lda_m8 <- lda(Loan_Status ~ Applicant_Income + Gender + Married + Education + Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
lda_m8.pred <- predict(lda_m8, newdata = test_set)
lda_m8.prob <- lda_m8.pred$posterior[, "1"]
lda_m8.trs = 0.80
lda_m8.pred_class<- ifelse(lda_m8.prob > lda_m8.trs, "Yes", "No")
lda_m8.nvector <- as.numeric(lda_m8.pred_class == "Yes")
lda_m8.fvector <- factor(lda_m8.nvector, levels = c(0, 1))
print(lda_m8.fvector)
confusionMatrix(data = lda_m8.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m8 <- prediction(lda_m8.prob,as.factor(test_set$Loan_Status))
perflda_m8 <- performance(predlda_m8,"tpr","fpr")
plot(perflda_m8, main="ROC curve for LDA", colorize=TRUE)
lda_m8_auc <- performance(predlda_m8, "auc")@y.values[[1]]
lda_m8_auc

#######################################################
#QDA M8
#######################################################
qda_m8 <- qda(Loan_Status ~ Applicant_Income + Gender + Married + Education + Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
qda_m8.pred <- predict(qda_m8, newdata = test_set)
qda_m8.prob <- qda_m8.pred$posterior[, "1"]
qda_m8.trs = 0.80
qda_m8.pred_class <- ifelse(qda_m8.prob > qda_m8.trs, "Yes", "No")
qda_m8.nvector <- as.numeric(qda_m8.pred_class == "Yes")
qda_m8.fvector <- factor(qda_m8.nvector, levels = c(0, 1))
print(qda_m8.fvector)
confusionMatrix(data = qda_m8.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m8 <- prediction(qda_m8.prob,as.factor(test_set$Loan_Status))
perfqda_m8 <- performance(predqda_m8,"tpr","fpr")
plot(perfqda_m8, main="ROC curve for QDA", colorize=TRUE)
qda_m8_auc <- performance(predqda_m8, "auc")@y.values[[1]]
qda_m8_auc

#######################################################
#Naive Bayes M8
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m8 <- naiveBayes(Loan_Status ~ Applicant_Income + Gender + Married + Education + Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
nb_m8.pred <- predict(nb_m8, newdata = test_set, type = "raw")
nb_m8.prob <- nb_m8.pred[, 2]
nb_m8.trs = 0.80
nb_m8.pred_class <- ifelse(nb_m8.prob > nb_m8.trs, "Yes", "No")
nb_m8.nvector <- as.numeric(nb_m8.pred_class == "Yes")
nb_m8.fvector <- factor(nb_m8.nvector, levels = c(0, 1))
print(nb_m8.fvector)
confusionMatrix(data = nb_m8.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m8 <- prediction(nb_m8.prob,as.factor(test_set$Loan_Status))
perfnb_m8 <- performance(prednb_m8,"tpr","fpr")
plot(perfnb_m8, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m8_auc <- performance(prednb_m8, "auc")@y.values[[1]]
nb_m8_auc

#######################################################
#Plot M8
#######################################################
plot(perflg_m8, main = "ROC curves m8", col = "blue", lwd = 2)
lines(perflda_m8@x.values[[1]], perflda_m8@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m8@x.values[[1]], perfqda_m8@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m8@x.values[[1]], perfnb_m8@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m8_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m8_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m8_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m8_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M9
#######################################################
logistic_m9 <- glm(Loan_Status ~ Applicant_Income + Gender + Married + Education + Education + Credit_History + Semiurban + Rural + Family_I_D, data = training_set, family = binomial)
summary(logistic_m9)
log_m9.test <- predict(logistic_m9, test_set, type="response")
log_m9.pred <- rep("No", nrow(test_set))
log_m9.pred[log_m9.test > 0.90] = "Yes"
log_nvector_m9 <- as.numeric(log_m9.pred == "Yes")
log_fvector_m9 <- factor(log_nvector_m9, levels = c(0, 1))
print(log_fvector_m9)
confusionMatrix(data=as.factor(log_nvector_m9),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m9 <- prediction(log_m9.test,as.factor(test_set$Loan_Status))
perflg_m9 <- performance(predlg_m9,"tpr","fpr")
plot(perflg_m9, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m9_auc<- performance(predlg_m9, "auc")@y.values[[1]]
log_m9_auc

#######################################################
#LDA M9
#######################################################
lda_m9 <- lda(Loan_Status ~ Applicant_Income + Gender + Married + Education + Education + Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
lda_m9.pred <- predict(lda_m9, newdata = test_set)
lda_m9.prob <- lda_m9.pred$posterior[, "1"]
lda_m9.trs = 0.90
lda_m9.pred_class<- ifelse(lda_m9.prob > lda_m9.trs, "Yes", "No")
lda_m9.nvector <- as.numeric(lda_m9.pred_class == "Yes")
lda_m9.fvector <- factor(lda_m9.nvector, levels = c(0, 1))
print(lda_m9.fvector)
confusionMatrix(data = lda_m9.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m9 <- prediction(lda_m9.prob,as.factor(test_set$Loan_Status))
perflda_m9 <- performance(predlda_m9,"tpr","fpr")
plot(perflda_m9, main="ROC curve for LDA", colorize=TRUE)
lda_m9_auc <- performance(predlda_m9, "auc")@y.values[[1]]
lda_m9_auc

#######################################################
#QDA M9
#######################################################
qda_m9 <- qda(Loan_Status ~ Applicant_Income + Gender + Married + Education + Education + Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
qda_m9.pred <- predict(qda_m9, newdata = test_set)
qda_m9.prob <- qda_m9.pred$posterior[, "1"]
qda_m9.trs = 0.90
qda_m9.pred_class <- ifelse(qda_m9.prob > qda_m9.trs, "Yes", "No")
qda_m9.nvector <- as.numeric(qda_m9.pred_class == "Yes")
qda_m9.fvector <- factor(qda_m9.nvector, levels = c(0, 1))
print(qda_m9.fvector)
confusionMatrix(data = qda_m9.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m9 <- prediction(qda_m9.prob,as.factor(test_set$Loan_Status))
perfqda_m9 <- performance(predqda_m9,"tpr","fpr")
plot(perfqda_m9, main="ROC curve for QDA", colorize=TRUE)
qda_m9_auc <- performance(predqda_m9, "auc")@y.values[[1]]
qda_m9_auc

#######################################################
#Naive Bayes M9
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m9 <- naiveBayes(Loan_Status ~ Applicant_Income + Gender + Married + Education + Education + Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
nb_m9.pred <- predict(nb_m9, newdata = test_set, type = "raw")
nb_m9.prob <- nb_m9.pred[, 2]
nb_m9.trs = 0.90
nb_m9.pred_class <- ifelse(nb_m9.prob > nb_m9.trs, "Yes", "No")
nb_m9.nvector <- as.numeric(nb_m9.pred_class == "Yes")
nb_m9.fvector <- factor(nb_m9.nvector, levels = c(0, 1))
print(nb_m9.fvector)
confusionMatrix(data = nb_m9.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m9 <- prediction(nb_m9.prob,as.factor(test_set$Loan_Status))
perfnb_m9 <- performance(prednb_m9,"tpr","fpr")
plot(perfnb_m9, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m9_auc <- performance(prednb_m9, "auc")@y.values[[1]]
nb_m9_auc

#######################################################
#Plot M9
#######################################################
plot(perflg_m9, main = "ROC curves m9", col = "blue", lwd = 2)
lines(perflda_m9@x.values[[1]], perflda_m9@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m9@x.values[[1]], perfqda_m9@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m9@x.values[[1]], perfnb_m9@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m9_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m9_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m9_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m9_auc, 3)), col = "purple", adj = 0, cex = 0.8)





#######################################################
#######################################################
#MODEL NUMBER 2 - UNDERSAMPLING#
#######################################################
#######################################################

#######################################################
#Dataset Loading
#######################################################
rm(list=ls())
getwd()
setwd("C:\\Users\\thoma\\OneDrive - unibs.it\\Statistical Learning\\Project\\Project_CT_726582")
dataset <- read.csv("loan_data_formatted3U.csv")

#Defining Training and Test Sets:
n <- nrow(dataset)
set.seed(0) #Set seed to ensure reproducibility 
train.ind <- sample(1:n, size = 0.75*n)

#The Training set will consist of 75% of the records and will be used for data analysis and exploration.
training_set <- dataset[train.ind,]
nrow(training_set) #125 observations

#The Test set will consist of the remaining 25% and will be used to test the various 
#models addressed during the course with respective evaluation of the results.
test_set <- dataset[-train.ind,]
nrow(test_set) #42 observations

#Count
count_d <- dataset %>%
  count(Loan_Status)
print(count_d) #Rejected = 86 and Accepted = 81

#Logistic Regression: Suitable for binary classification problems with linear relationships between predictors and the log-odds of the response.
#LDA: Suitable for classification with multivariate normal distributions and same covariance matrices for all classes.
#QDA: Suitable for classification with multivariate normal distributions and different covariance matrices for each class.
#Naive Bayes: Suitable for classification with the assumption of independence between features, simple and efficient, but may not be accurate if features are correlated.

#######################################################
#Logistic Regression M1
#######################################################
logistic_m1 <- glm(Loan_Status ~ Credit_History, data = training_set, family = binomial)
summary(logistic_m1)
log_m1.test <- predict(logistic_m1, test_set, type="response")
log_m1.pred <- rep("No", nrow(test_set))
log_m1.pred[log_m1.test > 0.60] = "Yes"
log_nvector_m1 <- as.numeric(log_m1.pred == "Yes")
log_fvector_m1 <- factor(log_nvector_m1, levels = c(0, 1))
print(log_fvector_m1)
confusionMatrix(data=as.factor(log_nvector_m1),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m1 <- prediction(log_m1.test,as.factor(test_set$Loan_Status))
perflg_m1 <- performance(predlg_m1,"tpr","fpr")
plot(perflg_m1, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m1_auc<- performance(predlg_m1, "auc")@y.values[[1]]
log_m1_auc

#######################################################
#LDA M1
#######################################################
lda_m1 <- lda(Loan_Status ~ Credit_History, data = training_set)
lda_m1.pred <- predict(lda_m1, newdata = test_set)
lda_m1.prob <- lda_m1.pred$posterior[, "1"]
lda_m1.trs = 0.70
lda_m1.pred_class<- ifelse(lda_m1.prob > lda_m1.trs, "Yes", "No")
lda_m1.nvector <- as.numeric(lda_m1.pred_class == "Yes")
lda_m1.fvector <- factor(lda_m1.nvector, levels = c(0, 1))
print(lda_m1.fvector)
confusionMatrix(data = lda_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m1 <- prediction(lda_m1.prob,as.factor(test_set$Loan_Status))
perflda_m1 <- performance(predlda_m1,"tpr","fpr")
plot(perflda_m1, main="ROC curve for LDA", colorize=TRUE)
lda_m1_auc <- performance(predlda_m1, "auc")@y.values[[1]]
lda_m1_auc

#######################################################
#QDA M1
#######################################################
qda_m1 <- qda(Loan_Status ~ Credit_History, data = training_set)
qda_m1.pred <- predict(qda_m1, newdata = test_set)
qda_m1.prob <- qda_m1.pred$posterior[, "1"]
qda_m1.trs = 0.60
qda_m1.pred_class <- ifelse(qda_m1.prob > qda_m1.trs, "Yes", "No")
qda_m1.nvector <- as.numeric(qda_m1.pred_class == "Yes")
qda_m1.fvector <- factor(qda_m1.nvector, levels = c(0, 1))
print(qda_m1.fvector)
confusionMatrix(data = qda_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m1 <- prediction(qda_m1.prob,as.factor(test_set$Loan_Status))
perfqda_m1 <- performance(predqda_m1,"tpr","fpr")
plot(perfqda_m1, main="ROC curve for QDA", colorize=TRUE)
qda_m1_auc <- performance(predqda_m1, "auc")@y.values[[1]]
qda_m1_auc

#######################################################
#Naive Bayes M1
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m1 <- naiveBayes(Loan_Status ~ Credit_History, data = training_set)
nb_m1.pred <- predict(nb_m1, newdata = test_set, type = "raw")
nb_m1.prob <- nb_m1.pred[, 2]
nb_m1.trs = 0.60
nb_m1.pred_class <- ifelse(nb_m1.prob > nb_m1.trs, "Yes", "No")
nb_m1.nvector <- as.numeric(nb_m1.pred_class == "Yes")
nb_m1.fvector <- factor(nb_m1.nvector, levels = c(0, 1))
print(nb_m1.fvector)
confusionMatrix(data = nb_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m1 <- prediction(nb_m1.prob,as.factor(test_set$Loan_Status))
perfnb_m1 <- performance(prednb_m1,"tpr","fpr")
plot(perfnb_m1, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m1_auc <- performance(prednb_m1, "auc")@y.values[[1]]
nb_m1_auc

#######################################################
#Plot M1
#######################################################
plot(perflg_m1, main = "ROC curves M1", col = "blue", lwd = 2)
lines(perflda_m1@x.values[[1]], perflda_m1@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m1@x.values[[1]], perfqda_m1@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m1@x.values[[1]], perfnb_m1@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m1_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m1_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m1_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m1_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M2
#######################################################
logistic_m2 <- glm(Loan_Status ~ Credit_History + Semiurban, data = training_set, family = binomial)
summary(logistic_m2)
log_m2.test <- predict(logistic_m2, test_set, type="response")
log_m2.pred <- rep("No", nrow(test_set))
log_m2.pred[log_m2.test > 0.40] = "Yes"
log_nvector_m2 <- as.numeric(log_m2.pred == "Yes")
log_fvector_m2 <- factor(log_nvector_m2, levels = c(0, 1))
print(log_fvector_m2)
confusionMatrix(data=as.factor(log_nvector_m2),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m2 <- prediction(log_m2.test,as.factor(test_set$Loan_Status))
perflg_m2 <- performance(predlg_m2,"tpr","fpr")
plot(perflg_m2, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m2_auc<- performance(predlg_m2, "auc")@y.values[[1]]
log_m2_auc

#######################################################
#LDA M2
#######################################################
lda_m2 <- lda(Loan_Status ~ Credit_History + Semiurban, data = training_set)
lda_m2.pred <- predict(lda_m2, newdata = test_set)
lda_m2.prob <- lda_m2.pred$posterior[, "1"]
lda_m2.trs = 0.65
lda_m2.pred_class<- ifelse(lda_m2.prob > lda_m2.trs, "Yes", "No")
lda_m2.nvector <- as.numeric(lda_m2.pred_class == "Yes")
lda_m2.fvector <- factor(lda_m2.nvector, levels = c(0, 1))
print(lda_m2.fvector)
confusionMatrix(data = lda_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m2 <- prediction(lda_m2.prob,as.factor(test_set$Loan_Status))
perflda_m2 <- performance(predlda_m2,"tpr","fpr")
plot(perflda_m2, main="ROC curve for LDA", colorize=TRUE)
lda_m2_auc <- performance(predlda_m2, "auc")@y.values[[1]]
lda_m2_auc

#######################################################
#QDA M2
#######################################################
qda_m2 <- qda(Loan_Status ~ Credit_History + Semiurban, data = training_set)
qda_m2.pred <- predict(qda_m2, newdata = test_set)
qda_m2.prob <- qda_m2.pred$posterior[, "1"]
qda_m2.trs = 0.65
qda_m2.pred_class <- ifelse(qda_m2.prob > qda_m2.trs, "Yes", "No")
qda_m2.nvector <- as.numeric(qda_m2.pred_class == "Yes")
qda_m2.fvector <- factor(qda_m2.nvector, levels = c(0, 1))
print(qda_m2.fvector)
confusionMatrix(data = qda_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m2 <- prediction(qda_m2.prob,as.factor(test_set$Loan_Status))
perfqda_m2 <- performance(predqda_m2,"tpr","fpr")
plot(perfqda_m2, main="ROC curve for QDA", colorize=TRUE)
qda_m2_auc <- performance(predqda_m2, "auc")@y.values[[1]]
qda_m2_auc

#######################################################
#Naive Bayes M2
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m2 <- naiveBayes(Loan_Status ~ Credit_History + Semiurban, data = training_set)
nb_m2.pred <- predict(nb_m2, newdata = test_set, type = "raw")
nb_m2.prob <- nb_m2.pred[, 2]
nb_m2.trs = 0.63
nb_m2.pred_class <- ifelse(nb_m2.prob > nb_m2.trs, "Yes", "No")
nb_m2.nvector <- as.numeric(nb_m2.pred_class == "Yes")
nb_m2.fvector <- factor(nb_m2.nvector, levels = c(0, 1))
print(nb_m2.fvector)
confusionMatrix(data = nb_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m2 <- prediction(nb_m2.prob,as.factor(test_set$Loan_Status))
perfnb_m2 <- performance(prednb_m2,"tpr","fpr")
plot(perfnb_m2, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m2_auc <- performance(prednb_m2, "auc")@y.values[[1]]
nb_m2_auc

#######################################################
#Plot M2
#######################################################
plot(perflg_m2, main = "ROC curves m2", col = "blue", lwd = 2)
lines(perflda_m2@x.values[[1]], perflda_m2@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m2@x.values[[1]], perfqda_m2@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m2@x.values[[1]], perfnb_m2@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m2_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m2_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m2_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m2_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M3
#######################################################
logistic_m3 <- glm(Loan_Status ~ Credit_History + Semiurban + Family_I_D, data = training_set, family = binomial)
summary(logistic_m3)
log_m3.test <- predict(logistic_m3, test_set, type="response")
log_m3.pred <- rep("No", nrow(test_set))
log_m3.pred[log_m3.test > 0.35] = "Yes"
log_nvector_m3 <- as.numeric(log_m3.pred == "Yes")
log_fvector_m3 <- factor(log_nvector_m3, levels = c(0, 1))
print(log_fvector_m3)
confusionMatrix(data=as.factor(log_nvector_m3),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m3 <- prediction(log_m3.test,as.factor(test_set$Loan_Status))
perflg_m3 <- performance(predlg_m3,"tpr","fpr")
plot(perflg_m3, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m3_auc<- performance(predlg_m3, "auc")@y.values[[1]]
log_m3_auc

#######################################################
#LDA M3
#######################################################
lda_m3 <- lda(Loan_Status ~ Credit_History + Semiurban + Family_I_D, data = training_set)
lda_m3.pred <- predict(lda_m3, newdata = test_set)
lda_m3.prob <- lda_m3.pred$posterior[, "1"]
lda_m3.trs = 0.63
lda_m3.pred_class<- ifelse(lda_m3.prob > lda_m3.trs, "Yes", "No")
lda_m3.nvector <- as.numeric(lda_m3.pred_class == "Yes")
lda_m3.fvector <- factor(lda_m3.nvector, levels = c(0, 1))
print(lda_m3.fvector)
confusionMatrix(data = lda_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m3 <- prediction(lda_m3.prob,as.factor(test_set$Loan_Status))
perflda_m3 <- performance(predlda_m3,"tpr","fpr")
plot(perflda_m3, main="ROC curve for LDA", colorize=TRUE)
lda_m3_auc <- performance(predlda_m3, "auc")@y.values[[1]]
lda_m3_auc

#######################################################
#QDA M3
#######################################################
qda_m3 <- qda(Loan_Status ~ Credit_History + Semiurban + Family_I_D, data = training_set)
qda_m3.pred <- predict(qda_m3, newdata = test_set)
qda_m3.prob <- qda_m3.pred$posterior[, "1"]
qda_m3.trs = 0.65
qda_m3.pred_class <- ifelse(qda_m3.prob > qda_m3.trs, "Yes", "No")
qda_m3.nvector <- as.numeric(qda_m3.pred_class == "Yes")
qda_m3.fvector <- factor(qda_m3.nvector, levels = c(0, 1))
print(qda_m3.fvector)
confusionMatrix(data = qda_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m3 <- prediction(qda_m3.prob,as.factor(test_set$Loan_Status))
perfqda_m3 <- performance(predqda_m3,"tpr","fpr")
plot(perfqda_m3, main="ROC curve for QDA", colorize=TRUE)
qda_m3_auc <- performance(predqda_m3, "auc")@y.values[[1]]
qda_m3_auc

#######################################################
#Naive Bayes M3
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m3 <- naiveBayes(Loan_Status ~ Credit_History + Semiurban + Family_I_D, data = training_set)
nb_m3.pred <- predict(nb_m3, newdata = test_set, type = "raw")
nb_m3.prob <- nb_m3.pred[, 2]
nb_m3.trs = 0.20
nb_m3.pred_class <- ifelse(nb_m3.prob > nb_m3.trs, "Yes", "No")
nb_m3.nvector <- as.numeric(nb_m3.pred_class == "Yes")
nb_m3.fvector <- factor(nb_m3.nvector, levels = c(0, 1))
print(nb_m3.fvector)
confusionMatrix(data = nb_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m3 <- prediction(nb_m3.prob,as.factor(test_set$Loan_Status))
perfnb_m3 <- performance(prednb_m3,"tpr","fpr")
plot(perfnb_m3, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m3_auc <- performance(prednb_m3, "auc")@y.values[[1]]
nb_m3_auc

#######################################################
#Plot M3
#######################################################
plot(perflg_m3, main = "ROC curves m3", col = "blue", lwd = 2)
lines(perflda_m3@x.values[[1]], perflda_m3@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m3@x.values[[1]], perfqda_m3@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m3@x.values[[1]], perfnb_m3@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m3_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m3_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m3_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m3_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M4
#######################################################
logistic_m4 <- glm(Loan_Status ~ Credit_History + Semiurban + Rural + Family_I_D, data = training_set, family = binomial)
summary(logistic_m4)
log_m4.test <- predict(logistic_m4, test_set, type="response")
log_m4.pred <- rep("No", nrow(test_set))
log_m4.pred[log_m4.test > 0.60] = "Yes"
log_nvector_m4 <- as.numeric(log_m4.pred == "Yes")
log_fvector_m4 <- factor(log_nvector_m4, levels = c(0, 1))
print(log_fvector_m4)
confusionMatrix(data=as.factor(log_nvector_m4),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m4 <- prediction(log_m4.test,as.factor(test_set$Loan_Status))
perflg_m4 <- performance(predlg_m4,"tpr","fpr")
plot(perflg_m4, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m4_auc<- performance(predlg_m4, "auc")@y.values[[1]]
log_m4_auc

#######################################################
#LDA M4
#######################################################
lda_m4 <- lda(Loan_Status ~ Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
lda_m4.pred <- predict(lda_m4, newdata = test_set)
lda_m4.prob <- lda_m4.pred$posterior[, "1"]
lda_m4.trs = 0.60
lda_m4.pred_class<- ifelse(lda_m4.prob > lda_m4.trs, "Yes", "No")
lda_m4.nvector <- as.numeric(lda_m4.pred_class == "Yes")
lda_m4.fvector <- factor(lda_m4.nvector, levels = c(0, 1))
print(lda_m4.fvector)
confusionMatrix(data = lda_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m4 <- prediction(lda_m4.prob,as.factor(test_set$Loan_Status))
perflda_m4 <- performance(predlda_m4,"tpr","fpr")
plot(perflda_m4, main="ROC curve for LDA", colorize=TRUE)
lda_m4_auc <- performance(predlda_m4, "auc")@y.values[[1]]
lda_m4_auc

#######################################################
#QDA M4
#######################################################
qda_m4 <- qda(Loan_Status ~ Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
qda_m4.pred <- predict(qda_m4, newdata = test_set)
qda_m4.prob <- qda_m4.pred$posterior[, "1"]
qda_m4.trs = 0.60
qda_m4.pred_class <- ifelse(qda_m4.prob > qda_m4.trs, "Yes", "No")
qda_m4.nvector <- as.numeric(qda_m4.pred_class == "Yes")
qda_m4.fvector <- factor(qda_m4.nvector, levels = c(0, 1))
print(qda_m4.fvector)
confusionMatrix(data = qda_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m4 <- prediction(qda_m4.prob,as.factor(test_set$Loan_Status))
perfqda_m4 <- performance(predqda_m4,"tpr","fpr")
plot(perfqda_m4, main="ROC curve for QDA", colorize=TRUE)
qda_m4_auc <- performance(predqda_m4, "auc")@y.values[[1]]
qda_m4_auc

#######################################################
#Naive Bayes M4
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m4 <- naiveBayes(Loan_Status ~ Credit_History + Semiurban + Rural + Family_I_D, data = training_set)
nb_m4.pred <- predict(nb_m4, newdata = test_set, type = "raw")
nb_m4.prob <- nb_m4.pred[, 2]
nb_m4.trs = 0.60
nb_m4.pred_class <- ifelse(nb_m4.prob > nb_m4.trs, "Yes", "No")
nb_m4.nvector <- as.numeric(nb_m4.pred_class == "Yes")
nb_m4.fvector <- factor(nb_m4.nvector, levels = c(0, 1))
print(nb_m4.fvector)
confusionMatrix(data = nb_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m4 <- prediction(nb_m4.prob,as.factor(test_set$Loan_Status))
perfnb_m4 <- performance(prednb_m4,"tpr","fpr")
plot(perfnb_m4, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m4_auc <- performance(prednb_m4, "auc")@y.values[[1]]
nb_m4_auc

#######################################################
#Plot M4
#######################################################
plot(perflg_m4, main = "ROC curves m4", col = "blue", lwd = 2)
lines(perflda_m4@x.values[[1]], perflda_m4@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m4@x.values[[1]], perfqda_m4@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m4@x.values[[1]], perfnb_m4@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m4_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m4_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m4_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m4_auc, 3)), col = "purple", adj = 0, cex = 0.8)





#######################################################
#######################################################
#MODEL NUMBER 3 - OVERSAMPLING#
#######################################################
#######################################################

#######################################################
#Dataset Loading
#######################################################
rm(list=ls())
getwd()
setwd("C:\\Users\\thoma\\OneDrive - unibs.it\\Statistical Learning\\Project\\Project_CT_726582")
dataset <- read.csv("loan_data_formatted3O.csv")

#Defining Training and Test Sets:
n <- nrow(dataset)
set.seed(0) #Set seed to ensure reproducibility 
train.ind <- sample(1:n, size = 0.75*n)

#The Training set will consist of 75% of the records and will be used for data analysis and exploration.
training_set <- dataset[train.ind,]
nrow(training_set) #363 observations

#The Test set will consist of the remaining 25% and will be used to test the various 
#models addressed during the course with respective evaluation of the results.
test_set <- dataset[-train.ind,]
nrow(test_set) #121 observations

#Count
count_d <- dataset %>%
  count(Loan_Status)
print(count_d) #Rejected = 242 and Accepted = 242


#Logistic Regression: Suitable for binary classification problems with linear relationships between predictors and the log-odds of the response.
#LDA: Suitable for classification with multivariate normal distributions and same covariance matrices for all classes.
#QDA: Suitable for classification with multivariate normal distributions and different covariance matrices for each class.
#Naive Bayes: Suitable for classification with the assumption of independence between features, simple and efficient, but may not be accurate if features are correlated.

#######################################################
#Logistic Regression M1
#######################################################
logistic_m1 <- glm(Loan_Status ~ Credit_History, data = training_set, family = binomial)
summary(logistic_m1)
log_m1.test <- predict(logistic_m1, test_set, type="response")
log_m1.pred <- rep("No", nrow(test_set))
log_m1.pred[log_m1.test > 0.60] = "Yes"
log_nvector_m1 <- as.numeric(log_m1.pred == "Yes")
log_fvector_m1 <- factor(log_nvector_m1, levels = c(0, 1))
print(log_fvector_m1)
confusionMatrix(data=as.factor(log_nvector_m1),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m1 <- prediction(log_m1.test,as.factor(test_set$Loan_Status))
perflg_m1 <- performance(predlg_m1,"tpr","fpr")
plot(perflg_m1, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m1_auc<- performance(predlg_m1, "auc")@y.values[[1]]
log_m1_auc

#######################################################
#LDA M1
#######################################################
lda_m1 <- lda(Loan_Status ~ Credit_History, data = training_set)
lda_m1.pred <- predict(lda_m1, newdata = test_set)
lda_m1.prob <- lda_m1.pred$posterior[, "1"]
lda_m1.trs = 0.70
lda_m1.pred_class<- ifelse(lda_m1.prob > lda_m1.trs, "Yes", "No")
lda_m1.nvector <- as.numeric(lda_m1.pred_class == "Yes")
lda_m1.fvector <- factor(lda_m1.nvector, levels = c(0, 1))
print(lda_m1.fvector)
confusionMatrix(data = lda_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m1 <- prediction(lda_m1.prob,as.factor(test_set$Loan_Status))
perflda_m1 <- performance(predlda_m1,"tpr","fpr")
plot(perflda_m1, main="ROC curve for LDA", colorize=TRUE)
lda_m1_auc <- performance(predlda_m1, "auc")@y.values[[1]]
lda_m1_auc

#######################################################
#QDA M1
#######################################################
qda_m1 <- qda(Loan_Status ~ Credit_History, data = training_set)
qda_m1.pred <- predict(qda_m1, newdata = test_set)
qda_m1.prob <- qda_m1.pred$posterior[, "1"]
qda_m1.trs = 0.60
qda_m1.pred_class <- ifelse(qda_m1.prob > qda_m1.trs, "Yes", "No")
qda_m1.nvector <- as.numeric(qda_m1.pred_class == "Yes")
qda_m1.fvector <- factor(qda_m1.nvector, levels = c(0, 1))
print(qda_m1.fvector)
confusionMatrix(data = qda_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m1 <- prediction(qda_m1.prob,as.factor(test_set$Loan_Status))
perfqda_m1 <- performance(predqda_m1,"tpr","fpr")
plot(perfqda_m1, main="ROC curve for QDA", colorize=TRUE)
qda_m1_auc <- performance(predqda_m1, "auc")@y.values[[1]]
qda_m1_auc

#######################################################
#Naive Bayes M1
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m1 <- naiveBayes(Loan_Status ~ Credit_History, data = training_set)
nb_m1.pred <- predict(nb_m1, newdata = test_set, type = "raw")
nb_m1.prob <- nb_m1.pred[, 2]
nb_m1.trs = 0.60
nb_m1.pred_class <- ifelse(nb_m1.prob > nb_m1.trs, "Yes", "No")
nb_m1.nvector <- as.numeric(nb_m1.pred_class == "Yes")
nb_m1.fvector <- factor(nb_m1.nvector, levels = c(0, 1))
print(nb_m1.fvector)
confusionMatrix(data = nb_m1.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m1 <- prediction(nb_m1.prob,as.factor(test_set$Loan_Status))
perfnb_m1 <- performance(prednb_m1,"tpr","fpr")
plot(perfnb_m1, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m1_auc <- performance(prednb_m1, "auc")@y.values[[1]]
nb_m1_auc

#######################################################
#Plot M1
#######################################################
plot(perflg_m1, main = "ROC curves M1", col = "blue", lwd = 2)
lines(perflda_m1@x.values[[1]], perflda_m1@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m1@x.values[[1]], perfqda_m1@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m1@x.values[[1]], perfnb_m1@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m1_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m1_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m1_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m1_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M2
#######################################################
logistic_m2 <- glm(Loan_Status ~ Credit_History + Semiurban, data = training_set, family = binomial)
summary(logistic_m2)
log_m2.test <- predict(logistic_m2, test_set, type="response")
log_m2.pred <- rep("No", nrow(test_set))
log_m2.pred[log_m2.test > 0.49] = "Yes"
log_nvector_m2 <- as.numeric(log_m2.pred == "Yes")
log_fvector_m2 <- factor(log_nvector_m2, levels = c(0, 1))
print(log_fvector_m2)
confusionMatrix(data=as.factor(log_nvector_m2),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m2 <- prediction(log_m2.test,as.factor(test_set$Loan_Status))
perflg_m2 <- performance(predlg_m2,"tpr","fpr")
plot(perflg_m2, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m2_auc<- performance(predlg_m2, "auc")@y.values[[1]]
log_m2_auc

#######################################################
#LDA M2
#######################################################
lda_m2 <- lda(Loan_Status ~ Credit_History + Semiurban, data = training_set)
lda_m2.pred <- predict(lda_m2, newdata = test_set)
lda_m2.prob <- lda_m2.pred$posterior[, "1"]
lda_m2.trs = 0.65
lda_m2.pred_class<- ifelse(lda_m2.prob > lda_m2.trs, "Yes", "No")
lda_m2.nvector <- as.numeric(lda_m2.pred_class == "Yes")
lda_m2.fvector <- factor(lda_m2.nvector, levels = c(0, 1))
print(lda_m2.fvector)
confusionMatrix(data = lda_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m2 <- prediction(lda_m2.prob,as.factor(test_set$Loan_Status))
perflda_m2 <- performance(predlda_m2,"tpr","fpr")
plot(perflda_m2, main="ROC curve for LDA", colorize=TRUE)
lda_m2_auc <- performance(predlda_m2, "auc")@y.values[[1]]
lda_m2_auc

#######################################################
#QDA M2
#######################################################
qda_m2 <- qda(Loan_Status ~ Credit_History + Semiurban, data = training_set)
qda_m2.pred <- predict(qda_m2, newdata = test_set)
qda_m2.prob <- qda_m2.pred$posterior[, "1"]
qda_m2.trs = 0.65
qda_m2.pred_class <- ifelse(qda_m2.prob > qda_m2.trs, "Yes", "No")
qda_m2.nvector <- as.numeric(qda_m2.pred_class == "Yes")
qda_m2.fvector <- factor(qda_m2.nvector, levels = c(0, 1))
print(qda_m2.fvector)
confusionMatrix(data = qda_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m2 <- prediction(qda_m2.prob,as.factor(test_set$Loan_Status))
perfqda_m2 <- performance(predqda_m2,"tpr","fpr")
plot(perfqda_m2, main="ROC curve for QDA", colorize=TRUE)
qda_m2_auc <- performance(predqda_m2, "auc")@y.values[[1]]
qda_m2_auc

#######################################################
#Naive Bayes M2
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m2 <- naiveBayes(Loan_Status ~ Credit_History + Semiurban, data = training_set)
nb_m2.pred <- predict(nb_m2, newdata = test_set, type = "raw")
nb_m2.prob <- nb_m2.pred[, 2]
nb_m2.trs = 0.63
nb_m2.pred_class <- ifelse(nb_m2.prob > nb_m2.trs, "Yes", "No")
nb_m2.nvector <- as.numeric(nb_m2.pred_class == "Yes")
nb_m2.fvector <- factor(nb_m2.nvector, levels = c(0, 1))
print(nb_m2.fvector)
confusionMatrix(data = nb_m2.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m2 <- prediction(nb_m2.prob,as.factor(test_set$Loan_Status))
perfnb_m2 <- performance(prednb_m2,"tpr","fpr")
plot(perfnb_m2, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m2_auc <- performance(prednb_m2, "auc")@y.values[[1]]
nb_m2_auc

#######################################################
#Plot M2
#######################################################
plot(perflg_m2, main = "ROC curves m2", col = "blue", lwd = 2)
lines(perflda_m2@x.values[[1]], perflda_m2@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m2@x.values[[1]], perfqda_m2@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m2@x.values[[1]], perfnb_m2@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m2_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m2_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m2_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m2_auc, 3)), col = "purple", adj = 0, cex = 0.8)



#######################################################
#Logistic Regression M3
#######################################################
logistic_m3 <- glm(Loan_Status ~ Married + Credit_History + Semiurban, data = training_set, family = binomial)
summary(logistic_m3)
log_m3.test <- predict(logistic_m3, test_set, type="response")
log_m3.pred <- rep("No", nrow(test_set))
log_m3.pred[log_m3.test > 0.50] = "Yes"
log_nvector_m3 <- as.numeric(log_m3.pred == "Yes")
log_fvector_m3 <- factor(log_nvector_m3, levels = c(0, 1))
print(log_fvector_m3)
confusionMatrix(data=as.factor(log_nvector_m3),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m3 <- prediction(log_m3.test,as.factor(test_set$Loan_Status))
perflg_m3 <- performance(predlg_m3,"tpr","fpr")
plot(perflg_m3, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m3_auc<- performance(predlg_m3, "auc")@y.values[[1]]
log_m3_auc

#######################################################
#LDA M3
#######################################################
lda_m3 <- lda(Loan_Status ~ Married + Credit_History + Semiurban, data = training_set)
lda_m3.pred <- predict(lda_m3, newdata = test_set)
lda_m3.prob <- lda_m3.pred$posterior[, "1"]
lda_m3.trs = 0.50
lda_m3.pred_class<- ifelse(lda_m3.prob > lda_m3.trs, "Yes", "No")
lda_m3.nvector <- as.numeric(lda_m3.pred_class == "Yes")
lda_m3.fvector <- factor(lda_m3.nvector, levels = c(0, 1))
print(lda_m3.fvector)
confusionMatrix(data = lda_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m3 <- prediction(lda_m3.prob,as.factor(test_set$Loan_Status))
perflda_m3 <- performance(predlda_m3,"tpr","fpr")
plot(perflda_m3, main="ROC curve for LDA", colorize=TRUE)
lda_m3_auc <- performance(predlda_m3, "auc")@y.values[[1]]
lda_m3_auc

#######################################################
#QDA M3
#######################################################
qda_m3 <- qda(Loan_Status ~ Married + Credit_History + Semiurban, data = training_set)
qda_m3.pred <- predict(qda_m3, newdata = test_set)
qda_m3.prob <- qda_m3.pred$posterior[, "1"]
qda_m3.trs = 0.50
qda_m3.pred_class <- ifelse(qda_m3.prob > qda_m3.trs, "Yes", "No")
qda_m3.nvector <- as.numeric(qda_m3.pred_class == "Yes")
qda_m3.fvector <- factor(qda_m3.nvector, levels = c(0, 1))
print(qda_m3.fvector)
confusionMatrix(data = qda_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m3 <- prediction(qda_m3.prob,as.factor(test_set$Loan_Status))
perfqda_m3 <- performance(predqda_m3,"tpr","fpr")
plot(perfqda_m3, main="ROC curve for QDA", colorize=TRUE)
qda_m3_auc <- performance(predqda_m3, "auc")@y.values[[1]]
qda_m3_auc

#######################################################
#Naive Bayes M3
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m3 <- naiveBayes(Loan_Status ~ Married + Credit_History + Semiurban, data = training_set)
nb_m3.pred <- predict(nb_m3, newdata = test_set, type = "raw")
nb_m3.prob <- nb_m3.pred[, 2]
nb_m3.trs = 0.50
nb_m3.pred_class <- ifelse(nb_m3.prob > nb_m3.trs, "Yes", "No")
nb_m3.nvector <- as.numeric(nb_m3.pred_class == "Yes")
nb_m3.fvector <- factor(nb_m3.nvector, levels = c(0, 1))
print(nb_m3.fvector)
confusionMatrix(data = nb_m3.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m3 <- prediction(nb_m3.prob,as.factor(test_set$Loan_Status))
perfnb_m3 <- performance(prednb_m3,"tpr","fpr")
plot(perfnb_m3, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m3_auc <- performance(prednb_m3, "auc")@y.values[[1]]
nb_m3_auc

#######################################################
#Plot M3
#######################################################
plot(perflg_m3, main = "ROC curves Model 3", col = "blue", lwd = 2)
lines(perflda_m3@x.values[[1]], perflda_m3@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m3@x.values[[1]], perfqda_m3@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m3@x.values[[1]], perfnb_m3@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m3_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m3_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m3_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m3_auc, 3)), col = "purple", adj = 0, cex = 0.8)


#######################################################
#Logistic Regression M4
#######################################################
logistic_m4 <- glm(Loan_Status ~ Married + Credit_History + Semiurban + Rural, data = training_set, family = binomial)
summary(logistic_m4)
log_m4.test <- predict(logistic_m4, test_set, type="response")
log_m4.pred <- rep("No", nrow(test_set))
log_m4.pred[log_m4.test > 0.65] = "Yes"
log_nvector_m4 <- as.numeric(log_m4.pred == "Yes")
log_fvector_m4 <- factor(log_nvector_m4, levels = c(0, 1))
print(log_fvector_m4)
confusionMatrix(data=as.factor(log_nvector_m4),reference=as.factor(test_set$Loan_Status),positive='1')
predlg_m4 <- prediction(log_m4.test,as.factor(test_set$Loan_Status))
perflg_m4 <- performance(predlg_m4,"tpr","fpr")
plot(perflg_m4, main="ROC curve for Logistic Regression", colorize=TRUE)
log_m4_auc<- performance(predlg_m4, "auc")@y.values[[1]]
log_m4_auc

#######################################################
#LDA M4
#######################################################
lda_m4 <- lda(Loan_Status ~ Married + Credit_History + Semiurban + Rural, data = training_set)
lda_m4.pred <- predict(lda_m4, newdata = test_set)
lda_m4.prob <- lda_m4.pred$posterior[, "1"]
lda_m4.trs = 0.10
lda_m4.pred_class<- ifelse(lda_m4.prob > lda_m4.trs, "Yes", "No")
lda_m4.nvector <- as.numeric(lda_m4.pred_class == "Yes")
lda_m4.fvector <- factor(lda_m4.nvector, levels = c(0, 1))
print(lda_m4.fvector)
confusionMatrix(data = lda_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predlda_m4 <- prediction(lda_m4.prob,as.factor(test_set$Loan_Status))
perflda_m4 <- performance(predlda_m4,"tpr","fpr")
plot(perflda_m4, main="ROC curve for LDA", colorize=TRUE)
lda_m4_auc <- performance(predlda_m4, "auc")@y.values[[1]]
lda_m4_auc

#######################################################
#QDA M4
#######################################################
qda_m4 <- qda(Loan_Status ~ Married + Credit_History + Semiurban + Rural, data = training_set)
qda_m4.pred <- predict(qda_m4, newdata = test_set)
qda_m4.prob <- qda_m4.pred$posterior[, "1"]
qda_m4.trs = 0.20
qda_m4.pred_class <- ifelse(qda_m4.prob > qda_m4.trs, "Yes", "No")
qda_m4.nvector <- as.numeric(qda_m4.pred_class == "Yes")
qda_m4.fvector <- factor(qda_m4.nvector, levels = c(0, 1))
print(qda_m4.fvector)
confusionMatrix(data = qda_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
predqda_m4 <- prediction(qda_m4.prob,as.factor(test_set$Loan_Status))
perfqda_m4 <- performance(predqda_m4,"tpr","fpr")
plot(perfqda_m4, main="ROC curve for QDA", colorize=TRUE)
qda_m4_auc <- performance(predqda_m4, "auc")@y.values[[1]]
qda_m4_auc

#######################################################
#Naive Bayes M4
#######################################################
training_set$Loan_Status <- as.factor(training_set$Loan_Status)
nb_m4 <- naiveBayes(Loan_Status ~ Married + Credit_History + Semiurban + Rural, data = training_set)
nb_m4.pred <- predict(nb_m4, newdata = test_set, type = "raw")
nb_m4.prob <- nb_m4.pred[, 2]
nb_m4.trs = 0.10
nb_m4.pred_class <- ifelse(nb_m4.prob > nb_m4.trs, "Yes", "No")
nb_m4.nvector <- as.numeric(nb_m4.pred_class == "Yes")
nb_m4.fvector <- factor(nb_m4.nvector, levels = c(0, 1))
print(nb_m4.fvector)
confusionMatrix(data = nb_m4.fvector, reference = as.factor(test_set$Loan_Status), positive = '1')
prednb_m4 <- prediction(nb_m4.prob,as.factor(test_set$Loan_Status))
perfnb_m4 <- performance(prednb_m4,"tpr","fpr")
plot(perfnb_m4, main="ROC curve for Naive Bayes", colorize=TRUE)
nb_m4_auc <- performance(prednb_m4, "auc")@y.values[[1]]
nb_m4_auc

#######################################################
#Plot M4
#######################################################
plot(perflg_m4, main = "ROC curves m4", col = "blue", lwd = 2)
lines(perflda_m4@x.values[[1]], perflda_m4@y.values[[1]], col = "red", lwd = 2)
lines(perfqda_m4@x.values[[1]], perfqda_m4@y.values[[1]], col = "green", lwd = 2)
lines(perfnb_m4@x.values[[1]], perfnb_m4@y.values[[1]], col = "purple", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "LDA", "QDA", "Naive Bayes"),
       col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8, box.lwd = 0.5)
text(0.65, 0.80, paste("AUC (Logistic):", round(log_m4_auc, 3)), col = "blue", adj = 0, cex = 0.8)
text(0.65, 0.75, paste("AUC (LDA):", round(lda_m4_auc, 3)), col = "red", adj = 0, cex = 0.8)
text(0.65, 0.70, paste("AUC (QDA):", round(qda_m4_auc, 3)), col = "green", adj = 0, cex = 0.8)
text(0.65, 0.65, paste("AUC (Naive Bayes):", round(nb_m4_auc, 3)), col = "purple", adj = 0, cex = 0.8)
