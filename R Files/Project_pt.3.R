#######################################################
#Statistical Learning Project - Pt.3
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
#Dataset Loading
#######################################################
getwd()
setwd("C:\\Users\\thoma\\OneDrive - unibs.it\\Statistical Learning\\Project\\Project_CT_726582")
dataset <- read.csv("loan_data_formatted3.csv")

#Defining Training and Test Sets:
n <- nrow(dataset)
set.seed(0) #Set seed to ensure reproducibility 
train.ind <- sample(1:n, size = 0.75*n)

#The Training set will consist of 75% of the records and will be used for data analysis and exploration.
training_set <- dataset[train.ind,]
nrow(training_set) #251 observations

#The Test set will consist of the remaining 25% and will be used to test the various 
#models addressed during the course with respective evaluation of the results.
test_set <- dataset[-train.ind,]
nrow(test_set) #84 observations

#Count
count_d <- dataset %>%
  count(Loan_Status)
print(count_d) #Rejected = 93 and Accepted = 242

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
log_m2.pred[log_m2.test > 0.80] = "Yes"
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
lda_m2.trs = 0.84
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
qda_m2.trs = 0.82
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
nb_m2.trs = 0.83
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
log_m3.pred[log_m3.test > 0.84] = "Yes"
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
lda_m3.trs = 0.87
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
qda_m3.trs = 0.86
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
nb_m3.trs = 0.86
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
log_m4.pred[log_m4.test > 0.82] = "Yes"
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
lda_m4.trs = 0.86
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
qda_m4.trs = 0.79
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



