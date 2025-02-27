#######################################################
#Statistical Learning Project - Pt.2 with balancing techniques
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
library(e1071)
library(ggthemes)
library(DMwR2)
library(ROCR)
library(performanceEstimation)
library(lattice)


#######################################################
#######################################################
#ANALYSIS NUMBER 1- SMOTE#
#######################################################
#######################################################

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

################################################################
#Unbalanced Data 1 - SMOTE
################################################################
b_set = dataset
sum(is.na(b_set$Loan_Status))
class(b_set$Loan_Status)
b_set$Loan_Status <- as.factor(b_set$Loan_Status)

set.seed(150)

new_set <- smote(Loan_Status ~ ., b_set, perc.over = 2, perc.under = 1.5)
#Spiegazione dell'uso dei parametri:
#perc.over: This parameter indicates the oversampling percentage for the minority class. A value of 2 means that each example of the minority class will be duplicated (200% oversampling).
#perc.under: This parameter indicates the undersampling percentage for the majority class. A value of 1.5 means that for each example of the minority class there will be 1.5 examples of the majority class (150% undersampling).
#This method tries to achieve a balance between classes by generating enough examples of the minority class and reducing the size of the majority class.
dim(new_set)
summary(new_set) #279 Rejected - 279 Accepted
new_set$Loan_Status

new_set$Loan_Status <- as.numeric(as.character(new_set$Loan_Status))
new_set <- new_set %>%
  mutate(across(c(Gender, Married, Education, Self_Employed, Credit_History, 
                  Loan_Status, Urban, Semiurban, Rural), 
                as.integer))
str(new_set)
summary(new_set)


#######################################################
#Correlation and Specific Tests
#######################################################
#Continuous Variables
continuous_vars <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 
                     'Loan_Amount', 'Loan_Amount_Term', 'Product_Income', 
                     'Family_Income', 'Family_I_D')

#Select only continuous variables from the new_set
d_correlation <- new_set[continuous_vars]
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
d_binary <- new_set[c('Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Urban', 'Semiurban', 'Rural', 'Loan_Status')]

#Chi-Square Test
chi_square_test <- function(var_bin) {
  tbl <- table(new_set$Loan_Status, new_set[[var_bin]])
  chi2 <- chisq.test(tbl)
  return(data.frame(Variable = var_bin, Chi_Square = chi2$statistic, p_value = chi2$p.value))
}
chi_square_results <- do.call(rbind, lapply(names(d_binary)[-length(names(d_binary))], chi_square_test))
chi_square_results <- chi_square_results[order(abs(chi_square_results$Chi_Square), decreasing = TRUE), ]
print(chi_square_results)
#All in all, almost all variables have interesting results, Self_Employed and Urban are but to a limited extent, Credit_History is incredibly significant.

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
  tbl <- table(new_set$Loan_Status, new_set[[var_bin]])
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
subset1 <- new_set[c('Applicant_Income', 'Coapplicant_Income','Loan_Status')]
#Partial Correlation
corr_partial1 <- pcor(subset1, method = "pearson")$estimate 
print(corr_partial1["Applicant_Income", "Loan_Status"])
print(corr_partial1["Coapplicant_Income", "Loan_Status"])
#Correlation
corr_value1 <- cor(new_set[["Applicant_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value1)
corr_value1. <- cor(new_set[["Coapplicant_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value1.) 
#Applicant_Income definitely better than Coapplicant_Income, however low.

subset2 <- new_set[c("Loan_Amount", "Loan_Amount_Term", "Loan_Status")]
#Partial Correlation
corr_partial2 <- pcor(subset2, method = "pearson")$estimate
print(corr_partial2["Loan_Amount", "Loan_Status"])
print(corr_partial2["Loan_Amount_Term", "Loan_Status"])
#Correlation
corr_value2 <- cor(new_set[["Loan_Amount"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value2)
corr_value2. <- cor(new_set[["Loan_Amount_Term"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value2.) 
#Partial correlation better than single correlation but still low.

subset3 <- new_set[c("Applicant_Income", "Product_Income", "Loan_Status")]
#Partial Correlation
corr_partial3 <- pcor(subset3, method = "pearson")$estimate
print(corr_partial3["Applicant_Income", "Loan_Status"])
print(corr_partial3["Product_Income", "Loan_Status"])
#Correlation
corr_value3 <- cor(new_set[["Applicant_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value3)
corr_value3. <- cor(new_set[["Product_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value3.) 
#Applicant_Income definitely better than Product_Income, however low.

Subset4 <- new_set[c("Coapplicant_Income", "Product_Income", "Loan_Status")]
#Partial Correlation
corr_partial4 <- pcor(Subset4, method = "pearson")$estimate
print(corr_partial4["Coapplicant_Income", "Loan_Status"])
print(corr_partial4["Product_Income", "Loan_Status"])
#Correlation
corr_value4 <- cor(new_set[["Coapplicant_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value4)
corr_value4. <- cor(new_set[["Product_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value4.) 
#Partial correlation better than single correlation but still low.

Subset5 <- new_set[c("Applicant_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial5 <- pcor(Subset5, method = "pearson")$estimate
print(corr_partial5["Applicant_Income", "Loan_Status"])
print(corr_partial5["Family_Income", "Loan_Status"])
#Correlation
corr_value5 <- cor(new_set[["Applicant_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value5)
corr_value5. <- cor(new_set[["Family_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value5.) 
#Single correlations better than partials, still low.

Subset6 <- new_set[c("Coapplicant_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial6 <- pcor(Subset6, method = "pearson")$estimate
print(corr_partial6["Coapplicant_Income", "Loan_Status"])
print(corr_partial6["Family_Income", "Loan_Status"])
#Correlation
corr_value6 <- cor(new_set[["Coapplicant_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value6)
corr_value6. <- cor(new_set[["Family_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value6.) 
#Partial correlation better than single correlation but still low.

Subset7 <- new_set[c("Product_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial7 <- pcor(Subset7, method = "pearson")$estimate
print(corr_partial7["Product_Income", "Loan_Status"])
print(corr_partial7["Family_Income", "Loan_Status"])
#Correlation
corr_value7 <- cor(new_set[["Product_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value7)
corr_value7. <- cor(new_set[["Family_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value7.) 
#In general very low values.

Subset8 <- new_set[c("Family_Income", "Dependents", "Loan_Status")]
#Partial Correlation
corr_partial8 <- pcor(Subset8, method = "pearson")$estimate
print(corr_partial8["Family_Income", "Loan_Status"])
print(corr_partial8["Dependents", "Loan_Status"])
#Correlation
corr_value8 <- cor(new_set[["Family_Income"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value8)
corr_value8. <- cor(new_set[["Dependents"]], new_set[["Loan_Status"]], use = "complete.obs")
print(corr_value8.) 
#Similar situation between partial and single but still low values.
#Those shown above are the combinations of greatest interest; however, the partial correlations of the other combinations were also checked.
#All in all, unfortunately, the partial correlation between the variables in respect to Y turns out to be generically low.

#Single Correlation
colnames(new_set)
var <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Loan_Status', 'Product_Income', 'Family_Income', 'Family_I_D')
var_corr <- sapply(var, function(variable) {
  cor(new_set[[variable]], new_set[["Loan_Status"]], use = "complete.obs")
})
v_dataframe <- data.frame(variable = var, correlation = var_corr)
print(v_dataframe)

#Eliminate continuous variables with correlation less than 0.1 (selected threshold)
v_threshold <- 0.1
s_var <- subset(v_dataframe, abs(var_corr) < v_threshold)
print(s_var)
all_var <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Loan_Status', 'Product_Income', 'Family_Income', 'Family_I_D')
corr_matrix1 <- cor(new_set[, all_var], use = "complete.obs")
print(corr_matrix1)
corr_LS <- corr_matrix1["Loan_Status", ]
print(corr_LS)
corr_LS <- corr_matrix1[, "Loan_Status"]
print(corr_LS)
var_LS <- names(corr_LS[abs(corr_LS) < v_threshold])
print(var_LS) 

#Continuous variables unfortunately have really too low a correlation so that in this case we are going to exclude all of them for model analysis.
#To these we also add the binaries with low significance obtained from the Chi-Square Test and Cramer's Index and keep only the remaining binaries.
new_set <- subset(new_set, select = -c(Dependents, Coapplicant_Income, Loan_Amount, Loan_Amount_Term, Urban, Product_Income, Family_Income))
str(new_set) #will then consist of 4 binary, 2 continuous and the Loan_Status

#VIF
#We recognise the limitations that Multicollinearity analysis has when binary or categorical variables are involved as in our case, 
#however having a number of statistical sources confirming that it is not overly problematic was chosen to report it.
vif1 <- vif(lm(Applicant_Income ~ Gender + Married + Education + Self_Employed + Credit_History +  Semiurban + Rural + Family_I_D, data = new_set))
print(vif1)
vif2 <- vif(lm(Gender ~ + Applicant_Income + Married + Education + Self_Employed + Credit_History +  Semiurban + Rural + Family_I_D, data = new_set))
print(vif2)
vif3 <- vif(lm(Married ~ Applicant_Income + Gender + Education + Self_Employed + Credit_History +  Semiurban + Rural + Family_I_D, data = new_set))
print(vif3)
vif4 <- vif(lm(Education ~ Applicant_Income + Gender + Married + Self_Employed + Credit_History +  Semiurban + Rural + Family_I_D, data = new_set))
print(vif4)
vif5 <- vif(lm(Self_Employed ~ Applicant_Income + Gender + Married + Education + Credit_History +  Semiurban + Rural + Family_I_D, data = new_set))
print(vif5)
vif6 <- vif(lm(Credit_History ~ Applicant_Income + Gender + Married + Education + Self_Employed +  Semiurban + Rural + Family_I_D, data = new_set))
print(vif6)
vif7 <- vif(lm(Semiurban ~ Applicant_Income + Gender + Married + Education + Self_Employed +  Credit_History + Rural + Family_I_D, data = new_set))
print(vif7)
vif8 <- vif(lm(Rural ~ Applicant_Income + Gender + Married + Education + Self_Employed +  Credit_History + Semiurban + Family_I_D, data = new_set))
print(vif8)
vif9 <- vif(lm(Family_I_D ~ Applicant_Income + Gender + Married + Education + Self_Employed +  Credit_History + Semiurban + Rural, data = new_set))
print(vif9)
#The results indicate that there is no significant collinearity between the variables in your dataset. 
#The VIFs are all very close to 1, which suggests that the independent variables in the model are not problematically correlated with each other.


#######################################################
#Outlier's Check
#######################################################
create_histogram_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_histogram(fill = 'seagreen', color = 'black', binwidth = 1) +  # Imposta il binwidth se necessario
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}
create_density_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_density(color = 'seagreen') +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}
plot_AI <- create_histogram_plot(new_set, "Applicant_Income", "Distribution of Applicant Income")
plot_G <- create_histogram_plot(new_set, "Gender", "Distribution of Gender")
plot_M <- create_histogram_plot(new_set, "Married", "Distribution of Married")
plot_E <- create_histogram_plot(new_set, "Education", "Distribution of Education")
plot_SE <- create_histogram_plot(new_set, "Self_Employed", "Distribution of Self_Employed")
plot_CH <- create_histogram_plot(new_set, "Credit_History", "Distribution of Credit History")
plot_SU <- create_histogram_plot(new_set, "Semiurban", "Distribution of Semiurban")
plot_R <- create_histogram_plot(new_set, "Rural", "Distribution of Rural")
plot_FID <- create_density_plot(new_set, "Family_I_D", "Distribution of Family_Income_Depedents")
grid.arrange(plot_AI, plot_G, plot_M, plot_E, plot_SE, plot_CH, plot_SU, plot_R, plot_FID, ncol = 2)

#Descriptive Statistics
s_var1 <- c('Applicant_Income', 'Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Semiurban', 'Rural', 'Family_I_D')
s_data <- new_set[s_var1]
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
  Applicant_Income = c(0.01, 0.99), #1% and 99% to remove extremely low or high values
  Gender = c(0, 1), 
  Married = c(0, 1), 
  Education = c(0, 1),
  Self_Employed = c(0, 1), 
  Credit_History = c(0, 1),
  Semiurban = c(0, 1), 
  Rural = c(0, 1), 
  Family_I_D =c(0.1, 0.9) #10% and 90% to remove extreme values, since the mean is much higher than the median
)
c_new_set <- new_set #clean

for (var in names(perc)) {
  lower_perc <- quantile(new_set[[var]], perc[[var]][1])
  upper_perc <- quantile(new_set[[var]], perc[[var]][2])
  c_new_set <- subset(c_new_set, c_new_set[[var]] >= lower_perc & c_new_set[[var]] <= upper_perc)
}
out <- anti_join(new_set, c_new_set) #outliers
num_out<- nrow(out)
num_c_new_set <- nrow(c_new_set)
print(paste("Number of discarded observations:", num_out)) #64
print(paste("Number of observations without outliers:", num_c_new_set)) #494
tot_obs = num_out + num_c_new_set
print(tot_obs)

create_histogram_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_histogram(fill = 'lightgray', color = 'black', binwidth = 1) +  
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}
create_density_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_density(color = 'lightgray') +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}
plot_AI1 <- create_histogram_plot(c_new_set, "Applicant_Income", "Distribution of Applicant Income")
plot_G1 <- create_histogram_plot(c_new_set, "Gender", "Distribution of Gender")
plot_M1 <- create_histogram_plot(c_new_set, "Married", "Distribution of Married")
plot_E1 <- create_histogram_plot(c_new_set, "Education", "Distribution of Education")
plot_SE1 <- create_histogram_plot(c_new_set, "Self_Employed", "Distribution of Self_Employed")
plot_CH1 <- create_histogram_plot(c_new_set, "Credit_History", "Distribution of Credit History")
plot_SU1 <- create_histogram_plot(c_new_set, "Semiurban", "Distribution of Semiurban")
plot_R1 <- create_histogram_plot(c_new_set, "Rural", "Distribution of Rural")
plot_FID1 <- create_density_plot(c_new_set, "Family_I_D", "Distribution of Family_Income_Depedents")
grid.arrange(plot_AI1, plot_G1, plot_M1, plot_E1, plot_SE1, plot_CH1, plot_SU1, plot_R1, plot_FID1, ncol = 2)

#Controllo sulla Y
hist1 <- ggplot(new_set, aes(x = Loan_Status)) +
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
subset_m <- regsubsets(Loan_Status ~ ., data = c_new_set, method = "exhaustive",nvmax = 15)
summary(subset_m)

#R2
summary(subset_m)$rss #RSS 
plot(summary(subset_m)$rss, xlab = "Number of Predictors", ylab = "Residual Sum of Squares", type = "b", col = "seagreen", pch=19)
summary(subset_m)$rsq #R2 
plot(summary(subset_m)$rsq, xlab = "Number of Predictors", ylab = "R2", type = "b", col = "black", pch=19)
m_adjR2 <- summary(subset_m)$adjr2
m_best_R2 <- which.max(m_adjR2)
plot(m_adjR2, xlab = "Number of Predictors", ylab = "Adjusted R2", type = "b", col = "black", pch = 19)
points(m_best_R2, m_adjR2[m_best_R2], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_R2)
par(cex.axis = 0.9, cex.lab = 0.9)
plot(subset_m, scale= "adjr2")
#The model with greater AdjR2 is the one composed of Married + Education + Credit_History + Semiurban + Rural.

#CP
m_cp <- summary(subset_m)$cp
m_best_cp <- which.min(m_cp)
plot(m_cp, xlab = "Number of Predictors", ylab = "Cp", type = "b", col = "black", pch = 19)
points(m_best_cp, m_cp[m_best_cp], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_cp)
plot(subset_m, scale= "Cp")
#The model with the minimum CP is generally considered the best of the models tested. 
#Therefore, the model composed of Married + Credit_History + Semiurban + Rural has a good balance between goodness of fit and complexity.

#BIC
m_bic <- summary(subset_m)$bic
m_best_bic <- which.min(m_bic)
plot(m_bic, xlab = "Number of Predictors", ylab = "BIC", type = "b", col = "black", pch = 19)
points(m_best_bic, m_bic[m_best_bic], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_bic)
plot(subset_m, scale= "bic")
#According to the BIC Test, the best model is the one consisting of Married + Credit_History + Semiurban.


#Save the modified data frame to a new CSV file
write.csv(c_new_set, file = "C:/Users/thoma/OneDrive - unibs.it/Statistical Learning/Project/Project_CT_726582/loan_data_formatted3S.csv", row.names = FALSE)





#######################################################
#######################################################
#ANALYSIS NUMBER 2 - UNDERSAMPLING#
#######################################################
#######################################################

#######################################################
#Dataset Loading 
#######################################################
rm(list=ls())
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


################################################################
#Unbalanced Data 2 - UNDERSAMPLING
################################################################
print(names(dataset))
head(dataset)
print(table(dataset$Loan_Status))

dataset$Loan_Status <- as.factor(dataset$Loan_Status)
str(dataset$Loan_Status)

library(caret)
#Undersampling
set.seed(150) 
undersampled_data <- downSample(x = dataset[, -which(names(dataset) == "Loan_Status")], 
                                y = dataset$Loan_Status,
                                yname = "Loan_Status")
print(table(undersampled_data$Loan_Status)) #93 Rejected and 93 Accepted 
head(undersampled_data)

undersampled_data$Loan_Status <- as.numeric(as.character(undersampled_data$Loan_Status))
undersampled_data <- undersampled_data  %>%
  mutate(Loan_Status = as.integer(Loan_Status))
str(undersampled_data)
summary(undersampled_data)

#######################################################
#Correlation and Specific Tests
#######################################################
#Continuous Variables
continuous_vars <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 
                     'Loan_Amount', 'Loan_Amount_Term', 'Product_Income', 
                     'Family_Income', 'Family_I_D')

#Select only continuous variables from the undersampled_data
d_correlation <- undersampled_data[continuous_vars]
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
d_binary <- undersampled_data[c('Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Urban', 'Semiurban', 'Rural', 'Loan_Status')]

#Chi-Square Test
chi_square_test <- function(var_bin) {
  tbl <- table(undersampled_data$Loan_Status, undersampled_data[[var_bin]])
  chi2 <- chisq.test(tbl)
  return(data.frame(Variable = var_bin, Chi_Square = chi2$statistic, p_value = chi2$p.value))
}
chi_square_results <- do.call(rbind, lapply(names(d_binary)[-length(names(d_binary))], chi_square_test))
chi_square_results <- chi_square_results[order(abs(chi_square_results$Chi_Square), decreasing = TRUE), ]
print(chi_square_results)
#Only Credit_History, Semiurban, and Rural remain highly signiﬁcant; the others unfortunately decline from before.

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
  tbl <- table(undersampled_data$Loan_Status, undersampled_data[[var_bin]])
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
#Situation analogous to the Chi-Square Test.

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
subset1 <- undersampled_data[c('Applicant_Income', 'Coapplicant_Income','Loan_Status')]
#Partial Correlation
corr_partial1 <- pcor(subset1, method = "pearson")$estimate 
print(corr_partial1["Applicant_Income", "Loan_Status"])
print(corr_partial1["Coapplicant_Income", "Loan_Status"])
#Correlation
corr_value1 <- cor(undersampled_data[["Applicant_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value1)
corr_value1. <- cor(undersampled_data[["Coapplicant_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value1.) 
#Simple correlation better than partial.

subset2 <- undersampled_data[c("Loan_Amount", "Loan_Amount_Term", "Loan_Status")]
#Partial Correlation
corr_partial2 <- pcor(subset2, method = "pearson")$estimate
print(corr_partial2["Loan_Amount", "Loan_Status"])
print(corr_partial2["Loan_Amount_Term", "Loan_Status"])
#Correlation
corr_value2 <- cor(undersampled_data[["Loan_Amount"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value2)
corr_value2. <- cor(undersampled_data[["Loan_Amount_Term"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value2.) 
#Partial correlation better than simple correlation

subset3 <- undersampled_data[c("Applicant_Income", "Product_Income", "Loan_Status")]
#Partial Correlation
corr_partial3 <- pcor(subset3, method = "pearson")$estimate
print(corr_partial3["Applicant_Income", "Loan_Status"])
print(corr_partial3["Product_Income", "Loan_Status"])
#Correlation
corr_value3 <- cor(undersampled_data[["Applicant_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value3)
corr_value3. <- cor(undersampled_data[["Product_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value3.) 
#Again, partial correlation better than simple correlation

Subset4 <- undersampled_data[c("Coapplicant_Income", "Product_Income", "Loan_Status")]
#Partial Correlation
corr_partial4 <- pcor(Subset4, method = "pearson")$estimate
print(corr_partial4["Coapplicant_Income", "Loan_Status"])
print(corr_partial4["Product_Income", "Loan_Status"])
#Correlation
corr_value4 <- cor(undersampled_data[["Coapplicant_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value4)
corr_value4. <- cor(undersampled_data[["Product_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value4.) 
#Partial correlation better than simple correlation

Subset5 <- undersampled_data[c("Applicant_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial5 <- pcor(Subset5, method = "pearson")$estimate
print(corr_partial5["Applicant_Income", "Loan_Status"])
print(corr_partial5["Family_Income", "Loan_Status"])
#Correlation
corr_value5 <- cor(undersampled_data[["Applicant_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value5)
corr_value5. <- cor(undersampled_data[["Family_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value5.) 
#Simple correlation better than partial.

Subset6 <- undersampled_data[c("Coapplicant_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial6 <- pcor(Subset6, method = "pearson")$estimate
print(corr_partial6["Coapplicant_Income", "Loan_Status"])
print(corr_partial6["Family_Income", "Loan_Status"])
#Correlation
corr_value6 <- cor(undersampled_data[["Coapplicant_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value6)
corr_value6. <- cor(undersampled_data[["Family_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value6.) 
#Partial correlation returns to better.

Subset7 <- undersampled_data[c("Product_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial7 <- pcor(Subset7, method = "pearson")$estimate
print(corr_partial7["Product_Income", "Loan_Status"])
print(corr_partial7["Family_Income", "Loan_Status"])
#Correlation
corr_value7 <- cor(undersampled_data[["Product_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value7)
corr_value7. <- cor(undersampled_data[["Family_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value7.) 
#Simple correlation better than partial.

Subset8 <- undersampled_data[c("Family_Income", "Dependents", "Loan_Status")]
#Partial Correlation
corr_partial8 <- pcor(Subset8, method = "pearson")$estimate
print(corr_partial8["Family_Income", "Loan_Status"])
print(corr_partial8["Dependents", "Loan_Status"])
#Correlation
corr_value8 <- cor(undersampled_data[["Family_Income"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value8)
corr_value8. <- cor(undersampled_data[["Dependents"]], undersampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value8.) 
#Simple correlation better than partial.
#Those shown above are the combinations of greatest interest; however, the partial correlations of the other combinations were also checked.
#All in all, unfortunately, the partial correlation between the variables in respect to Y turns out to be generically low.

#Single Correlation
colnames(undersampled_data)
var <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Loan_Status', 'Product_Income', 'Family_Income', 'Family_I_D')
var_corr <- sapply(var, function(variable) {
  cor(undersampled_data[[variable]], undersampled_data[["Loan_Status"]], use = "complete.obs")
})
v_dataframe <- data.frame(variable = var, correlation = var_corr)
print(v_dataframe)
#Most correlations are very low, suggesting that the numerical variables considered do not have a strong relationship with Loan_Status. 
#This could indicate that the numerical variables are not strong predictors of loan outcome.

#delete variables with correlation less than 0.1 (selected threshold)
v_threshold <- 0.1
s_var <- subset(v_dataframe, abs(var_corr) < v_threshold)
print(s_var)
all_var <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Loan_Status', 'Urban', 'Semiurban', 'Rural', 'Product_Income', 'Family_Income', 'Family_I_D')
corr_matrix1 <- cor(undersampled_data[, all_var], use = "complete.obs")
print(corr_matrix1)
corr_LS <- corr_matrix1["Loan_Status", ]
print(corr_LS)
corr_LS <- corr_matrix1[, "Loan_Status"]
print(corr_LS)
var_LS <- names(corr_LS[abs(corr_LS) < v_threshold])
print(var_LS)

#Continuous variables unfortunately have really too low a correlation so that in this case we are going to exclude all of them for model analysis.
#To these we also add the binaries with low significance obtained from the Chi-Square Test and Cramer's Index and keep only the remaining binaries.
undersampled_data <- subset(undersampled_data, select = -c(Dependents, Applicant_Income, Coapplicant_Income, Loan_Amount, Loan_Amount_Term, Gender, Married, Education, Self_Employed, Urban, Product_Income, Family_Income))
str(undersampled_data) #will then consist of 3 binaries, 1 continuous and Loan_Status

#VIF
#We recognise the limitations that Multicollinearity analysis has when binary or categorical variables are involved as in our case, 
#however having a number of statistical sources confirming that it is not overly problematic was chosen to report it.
vif1 <- vif(lm(Credit_History ~ Semiurban + Rural + Family_I_D, data = undersampled_data))
print(vif1)
vif2 <- vif(lm(Semiurban ~ Credit_History + Rural + Family_I_D, data = undersampled_data))
print(vif2)
vif3 <- vif(lm(Rural ~ Credit_History + Semiurban + Family_I_D, data = undersampled_data))
print(vif3)
vif4 <- vif(lm(Family_I_D ~ Credit_History + Semiurban + Rural, data = undersampled_data))
print(vif4)
#The results indicate that there is no significant collinearity between the variables in your dataset. 
#The VIFs are all very close to 1, which suggests that the independent variables in the model are not problematically correlated with each other.


#######################################################
#Outlier's Check
#######################################################
create_histogram_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_histogram(fill = 'seagreen', color = 'black', binwidth = 1) +  
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}
create_density_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_density(color = 'seagreen') +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}
plot_CH <- create_histogram_plot(undersampled_data, "Credit_History", "Distribution of Credit History")
plot_S <- create_histogram_plot(undersampled_data, "Semiurban", "Distribution of Semiurban")
plot_R <- create_histogram_plot(undersampled_data, "Rural", "Distribution of Rural")
plot_FID <- create_density_plot(undersampled_data, "Family_I_D", "Distribution of Family_Income_Depedents")
grid.arrange(plot_CH, plot_S, plot_R, plot_FID, ncol = 2)

#Descriptive Statistics
s_var1 <- c('Credit_History', 'Semiurban', 'Rural', 'Family_I_D')
s_data <- undersampled_data[s_var1]
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
  Credit_History = c(0, 1), 
  Semiurban = c(0, 1), 
  Rural = c(0, 1),
  Family_I_D =c(0.1, 0.9) #10% and 90% to remove extreme values, as the mean is much higher than the median
)
c_undersampled_data <- undersampled_data #clean

for (var in names(perc)) {
  lower_perc <- quantile(undersampled_data[[var]], perc[[var]][1])
  upper_perc <- quantile(undersampled_data[[var]], perc[[var]][2])
  c_undersampled_data <- subset(c_undersampled_data, c_undersampled_data[[var]] >= lower_perc & c_undersampled_data[[var]] <= upper_perc)
}
out <- anti_join(undersampled_data, c_undersampled_data) #outliers
num_out<- nrow(out)
num_c_undersampled_data <- nrow(c_undersampled_data)
print(paste("Number of discarded observations:", num_out)) #19
print(paste("Number of observations without outliers:", num_c_undersampled_data)) #167
tot_obs = num_out + num_c_undersampled_data 
print(tot_obs)

create_histogram_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_histogram(fill = 'lightgray', color = 'black', binwidth = 1) +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}
create_density_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable)) +
    geom_density(color = 'lightgray') +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}
plot_CH1 <- create_histogram_plot(c_undersampled_data, "Credit_History", "Distribution of Credit History")
plot_S1 <- create_histogram_plot(c_undersampled_data, "Semiurban", "Distribution of Semiurban")
plot_R1 <- create_histogram_plot(c_undersampled_data, "Rural", "Distribution of Rural")
plot_FID1 <- create_density_plot(c_undersampled_data, "Family_I_D", "Distribution of Family_Income_Depedents")
grid.arrange(plot_CH1, plot_S1, plot_R1, plot_FID1, ncol = 2)

#Controllo sulla Y
hist1 <- ggplot(undersampled_data, aes(x = Loan_Status)) +
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
subset_m <- regsubsets(Loan_Status ~ ., data = c_undersampled_data, method = "exhaustive",nvmax = 15)
summary(subset_m)

#R2
summary(subset_m)$rss #RSS 
plot(summary(subset_m)$rss, xlab = "Number of Predictors", ylab = "Residual Sum of Squares", type = "b", col = "seagreen", pch=19)
summary(subset_m)$rsq #R2 
plot(summary(subset_m)$rsq, xlab = "Number of Predictors", ylab = "R2", type = "b", col = "black", pch=19)
m_adjR2 <- summary(subset_m)$adjr2
m_best_R2 <- which.max(m_adjR2)
plot(m_adjR2, xlab = "Number of Predictors", ylab = "Adjusted R2", type = "b", col = "black", pch = 19)
points(m_best_R2, m_adjR2[m_best_R2], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_R2)
par(cex.axis = 0.9, cex.lab = 0.9)
plot(subset_m, scale= "adjr2")
#The model with greater AdjR2 is the one composed of Credit_History + Semiurban.

#CP
m_cp <- summary(subset_m)$cp
m_best_cp <- which.min(m_cp)
plot(m_cp, xlab = "Number of Predictors", ylab = "Cp", type = "b", col = "black", pch = 19)
points(m_best_cp, m_cp[m_best_cp], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_cp)
plot(subset_m, scale= "Cp")
#The model with the minimum CP is generally considered the best of the models tested. 
#Therefore, the model composed of Credit_History + Semiurban has a good balance between goodness of fit and complexity.

#BIC
m_bic <- summary(subset_m)$bic
m_best_bic <- which.min(m_bic)
plot(m_bic, xlab = "Number of Predictors", ylab = "BIC", type = "b", col = "black", pch = 19)
points(m_best_bic, m_bic[m_best_bic], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_bic)
plot(subset_m, scale= "bic")
#According to the BIC Test, the best model is the one consisting of Credit_History + Semiurban.


#Save the modified data frame to a new CSV file
write.csv(c_undersampled_data, file = "C:/Users/thoma/OneDrive - unibs.it/Statistical Learning/Project/Project_CT_726582/loan_data_formatted3U.csv", row.names = FALSE)





#######################################################
#######################################################
#ANALYSIS NUMBER 3 - OVERSAMPLING#
#######################################################
#######################################################

#######################################################
#Dataset Loading 
#######################################################
rm(list=ls())
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


################################################################
#Unbalanced Data 3 - OVERSAMPLING
################################################################ 
print(names(dataset))
head(dataset)
print(table(dataset$Loan_Status))

dataset$Loan_Status <- as.factor(dataset$Loan_Status)

library(caret)
#Oversampling
set.seed(150)
oversampled_data <- upSample(x = dataset[, -which(names(dataset) == "Loan_Status")],
                             y = dataset$Loan_Status,
                             yname = "Loan_Status")
print(table(oversampled_data$Loan_Status)) #242 Rejected and 242 Accepted

library(rlang) 
create_bar_plot <- function(data, variable, title) {
  var_sym <- sym(variable)
  ggplot(data, aes(x = !!var_sym, fill = !!var_sym)) +
    geom_bar() +
    geom_text(stat='count', aes(label=..count..), vjust=-0.5, color = 'black') +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab(variable) +
    ylab("Count") +
    scale_fill_manual(values = c('seagreen', 'seagreen3')) 
}
oversampled_data$Loan_Status <- as.factor(oversampled_data$Loan_Status)
plot_LS <- create_bar_plot(oversampled_data, "Loan_Status", "Distribution of Loan Status")
print(plot_LS)


oversampled_data$Loan_Status <- as.numeric(as.character(oversampled_data$Loan_Status))
oversampled_data <- oversampled_data %>%
  mutate(Loan_Status = as.integer(Loan_Status))
str(oversampled_data)
summary(oversampled_data)


#######################################################
#Correlation and Specific Tests
#######################################################
#Continuous Variables
continuous_vars <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 
                     'Loan_Amount', 'Loan_Amount_Term', 'Product_Income', 
                     'Family_Income', 'Family_I_D')

#Select only continuous variables from the oversampled_data
d_correlation <- oversampled_data[continuous_vars]
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
d_binary <- oversampled_data[c('Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Urban', 'Semiurban', 'Rural', 'Loan_Status')]

#Chi-Square Test
chi_square_test <- function(var_bin) {
  tbl <- table(oversampled_data$Loan_Status, oversampled_data[[var_bin]])
  chi2 <- chisq.test(tbl)
  return(data.frame(Variable = var_bin, Chi_Square = chi2$statistic, p_value = chi2$p.value))
}
chi_square_results <- do.call(rbind, lapply(names(d_binary)[-length(names(d_binary))], chi_square_test))
chi_square_results <- chi_square_results[order(abs(chi_square_results$Chi_Square), decreasing = TRUE), ]
print(chi_square_results)
#Overall, almost all variables have interesting results less than Urban and Self_Employed, Credit_History is incredibly significant.

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
  tbl <- table(oversampled_data$Loan_Status, oversampled_data[[var_bin]])
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
#Also here almost all variables have interesting results less than Self_Employed, Credit_History is incredibly significant.

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
subset1 <- oversampled_data[c('Applicant_Income', 'Coapplicant_Income','Loan_Status')]
#Partial Correlation
corr_partial1 <- pcor(subset1, method = "pearson")$estimate 
print(corr_partial1["Applicant_Income", "Loan_Status"])
print(corr_partial1["Coapplicant_Income", "Loan_Status"])
#Correlation
corr_value1 <- cor(oversampled_data[["Applicant_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value1)
corr_value1. <- cor(oversampled_data[["Coapplicant_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value1.) 
##In general very low correlations, but single is better than partial.

subset2 <- oversampled_data[c("Loan_Amount", "Loan_Amount_Term", "Loan_Status")]
#Partial Correlation
corr_partial2 <- pcor(subset2, method = "pearson")$estimate
print(corr_partial2["Loan_Amount", "Loan_Status"])
print(corr_partial2["Loan_Amount_Term", "Loan_Status"])
#Correlation
corr_value2 <- cor(oversampled_data[["Loan_Amount"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value2)
corr_value2. <- cor(oversampled_data[["Loan_Amount_Term"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value2.) 
#In this case, the partial proves to be slightly better.

subset3 <- oversampled_data[c("Applicant_Income", "Product_Income", "Loan_Status")]
#Partial Correlation
corr_partial3 <- pcor(subset3, method = "pearson")$estimate
print(corr_partial3["Applicant_Income", "Loan_Status"])
print(corr_partial3["Product_Income", "Loan_Status"])
#Correlation
corr_value3 <- cor(oversampled_data[["Applicant_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value3)
corr_value3. <- cor(oversampled_data[["Product_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value3.) 
#In this case, the partial proves to be slightly better.

Subset4 <- oversampled_data[c("Coapplicant_Income", "Product_Income", "Loan_Status")]
#Partial Correlation
corr_partial4 <- pcor(Subset4, method = "pearson")$estimate
print(corr_partial4["Coapplicant_Income", "Loan_Status"])
print(corr_partial4["Product_Income", "Loan_Status"])
#Correlation
corr_value4 <- cor(oversampled_data[["Coapplicant_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value4)
corr_value4. <- cor(oversampled_data[["Product_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value4.) 
#In this case, the partial proves to be much better.

Subset5 <- oversampled_data[c("Applicant_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial5 <- pcor(Subset5, method = "pearson")$estimate
print(corr_partial5["Applicant_Income", "Loan_Status"])
print(corr_partial5["Family_Income", "Loan_Status"])
#Correlation
corr_value5 <- cor(oversampled_data[["Applicant_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value5)
corr_value5. <- cor(oversampled_data[["Family_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value5.) 
#Very low but similar values

Subset6 <- oversampled_data[c("Coapplicant_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial6 <- pcor(Subset6, method = "pearson")$estimate
print(corr_partial6["Coapplicant_Income", "Loan_Status"])
print(corr_partial6["Family_Income", "Loan_Status"])
#Correlation
corr_value6 <- cor(oversampled_data[["Coapplicant_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value6)
corr_value6. <- cor(oversampled_data[["Family_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value6.) 
#Low values but with a particular difference between partial and single.

Subset7 <- oversampled_data[c("Product_Income", "Family_Income", "Loan_Status")]
#Partial Correlation
corr_partial7 <- pcor(Subset7, method = "pearson")$estimate
print(corr_partial7["Product_Income", "Loan_Status"])
print(corr_partial7["Family_Income", "Loan_Status"])
#Correlation
corr_value7 <- cor(oversampled_data[["Product_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value7)
corr_value7. <- cor(oversampled_data[["Family_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value7.) 
#Definitely better partial despite low values.

Subset8 <- oversampled_data[c("Family_Income", "Dependents", "Loan_Status")]
#Partial Correlation
corr_partial8 <- pcor(Subset8, method = "pearson")$estimate
print(corr_partial8["Family_Income", "Loan_Status"])
print(corr_partial8["Dependents", "Loan_Status"])
#Correlation
corr_value8 <- cor(oversampled_data[["Family_Income"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value8)
corr_value8. <- cor(oversampled_data[["Dependents"]], oversampled_data[["Loan_Status"]], use = "complete.obs")
print(corr_value8.) 
#Single better than partial.
#Those shown above are the combinations of greatest interest; however, the partial correlations of the other combinations were also checked.
#All in all, unfortunately, the partial correlation between the variables in respect to Y turns out to be generically low.

#Single Correlation
colnames(oversampled_data)
var <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Loan_Status', 'Product_Income', 'Family_Income', 'Family_I_D')
var_corr <- sapply(var, function(variable) {
  cor(oversampled_data[[variable]], oversampled_data[["Loan_Status"]], use = "complete.obs")
})
v_dataframe <- data.frame(variable = var, correlation = var_corr)
print(v_dataframe)
#Most correlations are very low, suggesting that the numerical variables considered do not have a strong relationship with Loan_Status. 
#This could indicate that the numeric variables are not strong predictors of loan outcome.

#delete variables with correlation less than 0.1 (selected threshold)
v_threshold <- 0.1
s_var <- subset(v_dataframe, abs(var_corr) < v_threshold)
print(s_var)
all_var <- c('Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Loan_Status', 'Urban', 'Semiurban', 'Rural', 'Product_Income', 'Family_Income', 'Family_I_D')
corr_matrix1 <- cor(oversampled_data[, all_var], use = "complete.obs")
print(corr_matrix1)
corr_LS <- corr_matrix1["Loan_Status", ]
print(corr_LS)
corr_LS <- corr_matrix1[, "Loan_Status"]
print(corr_LS)
var_LS <- names(corr_LS[abs(corr_LS) < v_threshold])
print(var_LS)

#Continuous variables unfortunately have really too low a correlation so that in this case we are going to exclude all of them for model analysis.
#To these we also add the binaries with low significance obtained from the Chi-Square Test and Cramer's Index and keep only the remaining binaries.
oversampled_data <- subset(oversampled_data, select = -c(Dependents, Applicant_Income, Coapplicant_Income, Loan_Amount, Loan_Amount_Term, Gender, Education, Self_Employed, Urban, Product_Income, Family_Income, Family_I_D))
str(oversampled_data) #will then consist of 4 binaries and Loan_Status

#Oversampling Dataset Composition
var_points <- data.frame(
  Variable = names(oversampled_data),
  Position = 1:length(names(oversampled_data))
)
ggplot(var_points, aes(x = Position, y = Variable)) +
  geom_point(size = 6, color = 'seagreen') +
  labs(x = "Index", y = "Variables", title = "Oversampling Model Composition") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 8),
    plot.title = element_text(hjust = 0.5)
  )

var_points <- var_points[var_points$Variable != "Rural", ]
library(kableExtra)
var_table <- data.frame(
  Variable = var_points$Variable
)
kable(var_table, align = 'c', col.names = "Variables in Model 3") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                full_width = F, 
                position = "center",
                fixed_thead = TRUE) %>%
  column_spec(1, bold = TRUE, color = "white", background = "seagreen") %>%
  kable_styling(position = "center", font_size = 14) %>%
  row_spec(0, bold = TRUE, font_size = 16)

#VIF
#We recognise the limitations that Multicollinearity analysis has when binary or categorical variables are involved as in our case, 
#however having a number of statistical sources confirming that it is not overly problematic was chosen to report it.
vif1 <- vif(lm(Married ~ Credit_History + Semiurban + Rural, data = oversampled_data))
print(vif1)
vif2 <- vif(lm(Credit_History ~ Married + Semiurban + Rural, data = oversampled_data))
print(vif2)
vif3 <- vif(lm(Semiurban ~ Married + Credit_History + Rural, data = oversampled_data))
print(vif3)
vif4 <- vif(lm(Rural ~ Married + Credit_History + Semiurban, data = oversampled_data))
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
plot_M <- create_bar_plot(oversampled_data, "Married", "Distribution of Married")
plot_CH <- create_bar_plot(oversampled_data, "Credit_History", "Distribution of Credit History")
plot_S <- create_bar_plot(oversampled_data, "Semiurban", "Distribution of Semiurban")
plot_R <- create_bar_plot(oversampled_data, "Rural", "Distribution of Rural")
grid.arrange(plot_M, plot_CH, plot_S, plot_R, ncol = 2)

#Descriptive Statistics
s_var1 <- c('Married', 'Credit_History', 'Semiurban', 'Rural')
s_data <- oversampled_data[s_var1]
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
c_oversampled_data <- oversampled_data #clean

for (var in names(perc)) {
  lower_perc <- quantile(oversampled_data[[var]], perc[[var]][1])
  upper_perc <- quantile(oversampled_data[[var]], perc[[var]][2])
  c_oversampled_data <- subset(c_oversampled_data, c_oversampled_data[[var]] >= lower_perc & c_oversampled_data[[var]] <= upper_perc)
}
out <- anti_join(oversampled_data, c_oversampled_data) #outliers
num_out<- nrow(out)
num_c_oversampled_data <- nrow(c_oversampled_data)
print(paste("Number of discarded observations:", num_out)) #0
print(paste("Number of observations without outliers:", num_c_oversampled_data)) #484
tot_obs = num_out + num_c_oversampled_data 
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
plot_M <- create_bar_plot(c_oversampled_data, "Married", "Distribution of Married")
plot_CH <- create_bar_plot(c_oversampled_data, "Credit_History", "Distribution of Credit History")
plot_S <- create_bar_plot(c_oversampled_data, "Semiurban", "Distribution of Semiurban")
plot_R <- create_bar_plot(c_oversampled_data, "Rural", "Distribution of Rural")
grid.arrange(plot_M, plot_CH, plot_S, plot_R, ncol = 2)


#Controllo sulla Y
hist1 <- ggplot(oversampled_data, aes(x = Loan_Status)) +
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
subset_m <- regsubsets(Loan_Status ~ ., data = oversampled_data, method = "exhaustive",nvmax = 15)
summary(subset_m)

#R2
summary(subset_m)$rss #RSS 
plot(summary(subset_m)$rss, xlab = "Number of Predictors", ylab = "Residual Sum of Squares", type = "b", col = "seagreen", pch=19)
summary(subset_m)$rsq #R2 
plot(summary(subset_m)$rsq, xlab = "Number of Predictors", ylab = "R2", type = "b", col = "black", pch=19)
m_adjR2 <- summary(subset_m)$adjr2
m_best_R2 <- which.max(m_adjR2)
plot(m_adjR2, xlab = "Number of Predictors", ylab = "Adjusted R2", type = "b", col = "black", pch = 19)
points(m_best_R2, m_adjR2[m_best_R2], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_R2)
par(cex.axis = 0.9, cex.lab = 0.9)
plot(subset_m, scale= "adjr2")
#The model with greater AdjR2 is the one composed of all variables.

#CP
m_cp <- summary(subset_m)$cp
m_best_cp <- which.min(m_cp)
plot(m_cp, xlab = "Number of Predictors", ylab = "Cp", type = "b", col = "black", pch = 19)
points(m_best_cp, m_cp[m_best_cp], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_cp)
plot(subset_m, scale= "Cp")
#The model with the minimum CP is generally considered the best of the models tested. 
#Therefore, the model composed of all variables has a good balance between goodness of fit and complexity.

#BIC
m_bic <- summary(subset_m)$bic
m_best_bic <- which.min(m_bic)
plot(m_bic, xlab = "Number of Predictors", ylab = "BIC", type = "b", col = "black", pch = 19)
points(m_best_bic, m_bic[m_best_bic], col = "seagreen", cex = 2, pch = 20)
coef(subset_m, m_best_bic)
plot(subset_m, scale= "bic")
#According to the BIC Test, the best model is the one consisting of Married + Credit_History + Semiurban.


#Save the modified data frame to a new CSV file
write.csv(oversampled_data, file = "C:/Users/thoma/OneDrive - unibs.it/Statistical Learning/Project/Project_CT_726582/loan_data_formatted3O.csv", row.names = FALSE)
