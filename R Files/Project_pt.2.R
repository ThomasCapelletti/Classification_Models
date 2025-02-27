#######################################################
#Statistical Learning Project - Pt.2
#Capelletti Thomas - 726582
#######################################################


#######################################################
#Delete all from the Environment and load the libraries
#######################################################
rm(list=ls())

library(dplyr)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(ppcor)
library(car)
library(MASS)
library(leaps)


#######################################################
#Dataset Loading
#######################################################
getwd()
setwd("C:\\Users\\thoma\\OneDrive - unibs.it\\Statistical Learning\\Project\\Project_CT_726582")
dataset <- read.csv("loan_data_formatted2.csv")
head(dataset)
class(dataset)
names(dataset)

colnames(dataset)[colnames(dataset) == "Gender_Male"] <- "Gender"
colnames(dataset)[colnames(dataset) == "Married_Yes"] <- "Married"
colnames(dataset)[colnames(dataset) == "Education_Not.Graduate"] <- "Education"
colnames(dataset)[colnames(dataset) == "Self_Employed_Yes"] <- "Self_Employed"
colnames(dataset)[colnames(dataset) == "Credit_History_1"] <- "Credit_History"
colnames(dataset)[colnames(dataset) == "Loan_Status_Y"] <- "Loan_Status"
colnames(dataset)[colnames(dataset) == "Property_Area_Urban"] <- "Urban"
colnames(dataset)[colnames(dataset) == "Property_Area_Semiurban"] <- "Semiurban"
colnames(dataset)[colnames(dataset) == "Property_Area_Rural"] <- "Rural"

#New variables
dataset <- dataset %>%
  mutate(Product_Income = Applicant_Income * Coapplicant_Income)
dataset <- dataset %>%
  mutate(Family_Income = Applicant_Income + Coapplicant_Income)
dataset <- dataset %>%
  mutate(Family_Income_Dependents = Family_Income * Dependents)
colnames(dataset)[colnames(dataset) == "Family_Income_Dependents"] <- "Family_I_D"


#######################################################
#Correlation and Specific Tests
#######################################################
#Continuous Variables
continuous_vars <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 
                     'Loan_Amount', 'Loan_Amount_Term', 'Product_Income', 
                     'Family_Income', 'Family_I_D')

#Select only continuous variables from the dataset
d_correlation <- dataset[continuous_vars]
d_correlation <- d_correlation %>%
  mutate_if(is.character, as.numeric)

#Calculation of the correlation matrix 
corr_matrix <- cor(d_correlation)

#Displaying the correlation matrix with the graph
corrplot(corr_matrix, 
         method = "color",          
         tl.col = "black",          
         addCoef.col = "black",     
         number.cex = 0.7,          
         tl.cex = 0.8,              
         type = "lower",            
         diag = TRUE,               
         order = "hclust",          
         addrect = 3,               
         outline = TRUE,            
         mar = c(0, 0, 2, 0))       

#Sorting and displaying the correlation matrix
corr_table <- as.data.frame(as.table(corr_matrix))
corr_table <- corr_table[order(abs(corr_table$Freq), decreasing = TRUE), ]
print(corr_table)


#Binary Variables
d_binary <- dataset[c('Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Urban', 'Semiurban', 'Rural', 'Loan_Status')]

#Chi-Square Test
chi_square_test <- function(var_bin) {
  tbl <- table(dataset$Loan_Status, dataset[[var_bin]])
  chi2 <- chisq.test(tbl)
  return(data.frame(Variable = var_bin, Chi_Square = chi2$statistic, p_value = chi2$p.value))
}
chi_square_results <- do.call(rbind, lapply(names(d_binary)[-length(names(d_binary))], chi_square_test))
chi_square_results <- chi_square_results[order(abs(chi_square_results$Chi_Square), decreasing = TRUE), ]
print(chi_square_results)
#Overall, almost all variables have interesting results less than Self_Employed, Credit_History is incredibly significant.

#Graphical Representation
ggplot(chi_square_results, aes(x = reorder(Variable, Chi_Square), y = Chi_Square, fill = Chi_Square)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Chi-Square Statistic by Variable",
       x = "Variable",
       y = "Chi-Square Statistic",
       fill = "Chi-Square") +
  scale_fill_gradient(low = "lightgray", high = "seagreen") +
  theme_minimal()

#Cramer Index
cramer_v <- function(var_bin) {
  tbl <- table(dataset$Loan_Status, dataset[[var_bin]])
  chi2 <- chisq.test(tbl)
  n <- sum(tbl)
  phi2 <- chi2$statistic / n
  r <- nrow(tbl)
  k <- ncol(tbl)
  V <- sqrt(phi2 / min(r - 1, k - 1))
  return(data.frame(Variable = var_bin, Cramer_V = V))
}
cramer_v_results <- do.call(rbind, lapply(names(d_binary)[-length(names(d_binary))], cramer_v))
cramer_v_results <- cramer_v_results[order(abs(cramer_v_results$Cramer_V), decreasing = TRUE), ]
print(cramer_v_results)
#Slightly different situation from the previous one, here some variables turn out not to be so significant, overall the main ones remain influential.

#Graphical Representation
ggplot(cramer_v_results, aes(x = reorder(Variable, Cramer_V), y = Cramer_V, fill = Cramer_V)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Cramer's V by Variable",
       x = "Variable",
       y = "Cramer's V",
       fill = "Cramer's V") +
  scale_fill_gradient(low = "lightgray", high = "seagreen") +
  theme_minimal()


#######################################################
#Correlation and Partial correlation
#######################################################
subset1 <- dataset[c('Applicant_Income', 'Coapplicant_Income','Loan_Status')]
#Partial Correlation
corr_partial1 <- pcor(subset1, method = "pearson")$estimate 
print(corr_partial1["Applicant_Income", "Loan_Status"])
print(corr_partial1["Coapplicant_Income", "Loan_Status"])
#Correlation
corr_value1 <- cor(dataset[["Applicant_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value1)
corr_value1. <- cor(dataset[["Coapplicant_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value1.) 
#In general very low correlations

subset2 <- dataset[c("Loan_Amount", "Loan_Amount_Term", "Loan_Status")]
#Partial Correlation
corr_partial2 <- pcor(subset2, method = "pearson")$estimate
print(corr_partial2["Loan_Amount", "Loan_Status"])
print(corr_partial2["Loan_Amount_Term", "Loan_Status"])
#Correlation
corr_value2 <- cor(dataset[["Loan_Amount"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2)
corr_value2. <- cor(dataset[["Loan_Amount_Term"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2.) 
#Partial correlation of Loan_Amount better than the rest

subset3 <- dataset[c("Applicant_Income", "Product_Income", "Loan_Status")]
#Partial Correlation
corr_partial2 <- pcor(subset3, method = "pearson")$estimate
print(corr_partial2["Applicant_Income", "Loan_Status"])
print(corr_partial2["Product_Income", "Loan_Status"])
#Correlation
corr_value2 <- cor(dataset[["Applicant_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2)
corr_value2. <- cor(dataset[["Product_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2.) 
#Again overall really very low correlations.

Subset4 <- dataset[c("Coapplicant_Income", "Product_Income", "Loan_Status")]
#Partial Correlation
corr_partial2 <- pcor(Subset4, method = "pearson")$estimate
print(corr_partial2["Coapplicant_Income", "Loan_Status"])
print(corr_partial2["Product_Income", "Loan_Status"])
#Correlation
corr_value2 <- cor(dataset[["Coapplicant_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2)
corr_value2. <- cor(dataset[["Product_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2.) 
#The Product_Income shows an excessively small correlation.

Subset5 <- dataset[c("Applicant_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial2 <- pcor(Subset5, method = "pearson")$estimate
print(corr_partial2["Applicant_Income", "Loan_Status"])
print(corr_partial2["Family_Income", "Loan_Status"])
#Correlation
corr_value2 <- cor(dataset[["Applicant_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2)
corr_value2. <- cor(dataset[["Family_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2.) 
#Very low values indeed.

Subset6 <- dataset[c("Coapplicant_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial2 <- pcor(Subset6, method = "pearson")$estimate
print(corr_partial2["Coapplicant_Income", "Loan_Status"])
print(corr_partial2["Family_Income", "Loan_Status"])
#Correlation
corr_value2 <- cor(dataset[["Coapplicant_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2)
corr_value2. <- cor(dataset[["Family_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value2.) 
##Very low values indeed.

Subset7 <- dataset[c("Product_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial5 <- pcor(Subset7, method = "pearson")$estimate
print(corr_partial5["Product_Income", "Loan_Status"])
print(corr_partial5["Family_Income", "Loan_Status"])
#Correlation
corr_value5 <- cor(dataset[["Product_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value5)
corr_value5. <- cor(dataset[["Family_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value5.) 
##Very low values indeed.

Subset8 <- dataset[c("Family_Income", "Dependents", "Loan_Status")]
#Partial Correlation
corr_partial6 <- pcor(Subset8, method = "pearson")$estimate
print(corr_partial6["Family_Income", "Loan_Status"])
print(corr_partial6["Dependents", "Loan_Status"])
#Correlation
corr_value6 <- cor(dataset[["Family_Income"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value6)
corr_value6. <- cor(dataset[["Dependents"]], dataset[["Loan_Status"]], use = "complete.obs")
print(corr_value6.) 
#Very low values indeed.
#Those shown above are the combinations of greatest interest; however, the partial correlations of the other combinations were also checked.
#All in all, unfortunately, the partial correlation between the variables in respect to Y turns out to be generically low.

#Single Correlation
colnames(dataset)
var <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Loan_Status', 'Product_Income', 'Family_Income', 'Family_I_D')
var_corr <- sapply(var, function(variable) {
  cor(dataset[[variable]], dataset[["Loan_Status"]], use = "complete.obs")
})
v_dataframe <- data.frame(variable = var, correlation = var_corr)
print(v_dataframe)
#Most correlations are very low, suggesting that the numerical variables considered do not have a strong relationship with Loan_Status. 
#This could indicate that the numeric variables are not strong predictors of loan outcome.

#Eliminate continuous variables with correlation less than 0.1 (selected threshold)
v_threshold <- 0.1
s_var <- subset(v_dataframe, abs(var_corr) < v_threshold)
print(s_var)
all_var <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Loan_Status', 'Product_Income', 'Family_Income', 'Family_I_D')
corr_matrix1 <- cor(dataset[, all_var], use = "complete.obs")
print(corr_matrix1)
corr_LS <- corr_matrix1["Loan_Status", ]
print(corr_LS)
corr_LS <- corr_matrix1[, "Loan_Status"]
print(corr_LS)
var_LS <- names(corr_LS[abs(corr_LS) < v_threshold])
print(var_LS) 

#Continuous variables unfortunately have really too low a correlation so that in this case we are going to exclude all of them for model analysis.
#To these we also add the binaries with low significance obtained from the Chi-Square Test and Cramer's Index and keep only the remaining binaries.
dataset <- subset(dataset, select = -c(Dependents, Applicant_Income, Coapplicant_Income, Loan_Amount, Loan_Amount_Term, Gender, Education, Self_Employed, Urban, Product_Income, Family_Income, Family_I_D))
str(dataset) #will then consist of 4 binaries and Loan_Status

#VIF
#We recognise the limitations that Multicollinearity analysis has when binary or categorical variables are involved as in our case, 
#however having a number of statistical sources confirming that it is not overly problematic was chosen to report it.
vif1 <- vif(lm(Married ~ Credit_History + Semiurban + Rural, data = dataset))
print(vif1)
vif2 <- vif(lm(Credit_History ~ Married + Semiurban + Rural, data = dataset))
print(vif2)
vif3 <- vif(lm(Semiurban ~ Married + Credit_History + Rural, data = dataset))
print(vif3)
vif4 <- vif(lm(Rural ~ Married + Credit_History + Semiurban, data = dataset))
print(vif4)
#The results indicate that there is no significant collinearity between the variables in your dataset. 
#The VIFs are all very close to 1, which suggests that the independent variables in the model are not problematically correlated with each other.


#######################################################
#Outlier's Check
#######################################################
create_bar_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_bar(fill = 'seagreen') +
    geom_text(stat='count', aes(label=..count..), vjust=-0.5, color = 'black') +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab(variable) +
    ylab("Count")
}
plot_M <- create_bar_plot(dataset, "Married", "Distribution of Married")
plot_CH <- create_bar_plot(dataset, "Credit_History", "Distribution of Credit History")
plot_S <- create_bar_plot(dataset, "Semiurban", "Distribution of Semiurban")
plot_R <- create_bar_plot(dataset, "Rural", "Distribution of Rural")
grid.arrange(plot_M, plot_CH, plot_S, plot_R, ncol = 2)

#Descriptive Statistics
s_var1 <- c('Married', 'Credit_History', 'Semiurban', 'Rural')
s_data <- dataset[s_var1]
s_data_sum <- data.frame(
  Variable = colnames(s_data),
  Min = sapply(s_data, min),
  Mean = sapply(s_data, mean),
  Median = sapply(s_data, median),
  Max = sapply(s_data, max)
)
print(s_data_sum)

#Outliers elimination
perc <- list(
  Married = c(0, 1), 
  Credit_History = c(0, 1), 
  Semiurban = c(0, 1), 
  Rural = c(0, 1)
)
c_dataset <- dataset #clean

for (var in names(perc)) {
  lower_perc <- quantile(dataset[[var]], perc[[var]][1])
  upper_perc <- quantile(dataset[[var]], perc[[var]][2])
  c_dataset <- subset(c_dataset, c_dataset[[var]] >= lower_perc & c_dataset[[var]] <= upper_perc)
}
out <- anti_join(dataset, c_dataset) #outliers
num_out<- nrow(out)
num_c_dataset <- nrow(c_dataset)
print(paste("Number of discarded observations:", num_out)) #0
print(paste("Number of observations without outliers:", num_c_dataset)) #335
tot_obs = num_out + num_c_dataset
print(tot_obs)

create_bar_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_bar(fill = 'lightgray') +
    geom_text(stat='count', aes(label=..count..), vjust=-0.5, color = 'black') +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab(variable) +
    ylab("Count")
}
plot_M1 <- create_bar_plot(c_dataset, "Married", "Distribution of Married")
plot_CH1 <- create_bar_plot(c_dataset, "Credit_History", "Distribution of Credit History")
plot_S1 <- create_bar_plot(c_dataset, "Semiurban", "Distribution of Semiurban")
plot_R1 <- create_bar_plot(c_dataset, "Rural", "Distribution of Rural")
grid.arrange(plot_M1, plot_CH1, plot_S1, plot_R1, ncol = 2)

#Controllo sulla Y
hist1 <- ggplot(dataset, aes(x = Loan_Status)) +
  geom_histogram(binwidth = 1, fill = "seagreen", color = "black", alpha = 0.7) +
  labs(title = "Loan Status", x = "Loan Status", y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5), axis.text = element_text(size = 10))+
  scale_y_continuous(labels = scales::comma)
hist2 <- ggplot(out, aes(x = Loan_Status)) +
  geom_histogram(binwidth = 1, fill = "lightgray", color = "black", alpha = 0.7) +
  labs(title = "Loan Status Outliers", x = "Loan", y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5), axis.text = element_text(size = 10))+
  scale_y_continuous(labels = scales::comma)
grid.arrange(hist1, hist2, ncol = 2)


#######################################################
#Choosing the Optimal Model
#######################################################
subset_m <- regsubsets(Loan_Status ~ ., data = dataset, method = "exhaustive",nvmax = 15)
summary(subset_m)

#R2
summary(subset_m)$rss #RSS 
plot(summary(subset_m)$rss, xlab = "Number of Predictors", ylab = "Residual Sum of Squares", type = "b", col = "black", pch=19)
summary(subset_m)$rsq #R2 
plot(summary(subset_m)$rsq, xlab = "Number of Predictors", ylab = "R2", type = "b", col = "black", pch=19)
m_adjR2 <- summary(subset_m)$adjr2
m_best_R2 <- which.max(m_adjR2)
plot(m_adjR2, xlab = "Number of Predictors", ylab = "Adjusted R2", type = "b", col = "black", pch = 19)
points(m_best_R2, m_adjR2[m_best_R2], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_R2)
par(cex.axis = 0.9, cex.lab = 0.9)
plot(subset_m, scale= "adjr2")
#The model with greater AdjR2 is the one composed of Married + Credit_History + Semiurban.

#CP
m_cp <- summary(subset_m)$cp
m_best_cp <- which.min(m_cp)
plot(m_cp, xlab = "Number of Predictors", ylab = "Cp", type = "b", col = "black", pch = 19)
points(m_best_cp, m_cp[m_best_cp], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_cp)
plot(subset_m, scale= "Cp")
#The model with the minimum CP is generally considered the best of the models tested. 
#Therefore, the model composed of Married + Credit_History + Semiurban has a good balance between goodness of fit and complexity.

#BIC
m_bic <- summary(subset_m)$bic
m_best_bic <- which.min(m_bic)
plot(m_bic, xlab = "Number of Predictors", ylab = "BIC", type = "b", col = "black", pch = 19)
points(m_best_bic, m_bic[m_best_bic], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_bic)
plot(subset_m, scale= "bic")
#According to the BIC Test, the best model is the one consisting of Married + Credit_History + Semiurban.


#Save the modified data frame to a new CSV file
write.csv(dataset, file = "C:/Users/thoma/OneDrive - unibs.it/Statistical Learning/Project/Project_CT_726582/loan_data_formatted3.csv", row.names = FALSE)

