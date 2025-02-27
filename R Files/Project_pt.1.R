#######################################################
#Statistical Learning Project - Pt.1
#Capelletti Thomas - 726582
#######################################################


#######################################################
#Preliminary Operations and Dataset Loading
#######################################################
rm(list=ls())
#Set the directory
setwd("C:\\Users\\thoma\\OneDrive - unibs.it\\Statistical Learning\\Project\\Project_CT_726582")

#Upload the necessary packages
library(tidyverse)
library(fastDummies)
library(gridExtra)

#Importing Data 
#Dataset obtained from 'https://www.kaggle.com/datasets/willianoliveiragibin/federal-trade-commission-ftc'
dataset <- read.csv("loan_data_formatted.csv",sep=";")

#Remove rows that contain at least one missing value
cleaned_dataset <- dataset %>% na.omit()
#Rimozione valori non coerenti
dataset <- cleaned_dataset %>%
  filter(Gender != 0, Married != 0, Self_Employed != 0, Loan_Amount_Term != 0) %>%
  filter(!Loan_ID %in% c("LP001915", "LP002369")) #In this case we eliminated by acting directly on the Loan_IDs since the entire rows were problematic.
#These deletions were made following a direct discussion with the creator of the dataset.

#Carried out a Data Cleaning operation by eliminating rows with such missing values as they were causing problems
#5 Gender
#21 Self Employed
#22 Loan Amount Term 
#1 Credit History 
#2 Coapplicant Income 
#The dataset originally contained 381 records, now contains 335 records

#Structure of the data frame and its main characteristics 
str(dataset) #335 observations and 13 variables
names(dataset) 
dim(dataset) #335 rows and 13 columns
summary(dataset) #There are variables of various kinds, run for more detail

#Renaming columns
old_labels = colnames(dataset)
new_col_names = c('Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 
                  'Self_Employed', 'Applicant_Income', 'Coapplicant_Income', 
                  'Loan_Amount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status')
colnames(dataset) = new_col_names


#######################################################
#Explorative Data analysis
#######################################################
data_graph <- dataset

#Transform all qualitative variables into factors to conduct a preliminary exploratory analysis.
#The initial objective is certainly to view the distributions of our variables from a graphical point of view.
data_graph$Loan_ID <- as.factor(data_graph$Loan_ID) #Loan_ID
data_graph$Gender <- factor(data_graph$Gender, levels = c("Male", "Female")) #Gender
data_graph$Dependents <- as.factor(data_graph$Dependents) #Dependents
data_graph$Education <- as.factor(data_graph$Education) #Education
data_graph$Self_Employed <- as.factor(data_graph$Self_Employed) #Self_Employed
data_graph$Credit_History <- as.factor(data_graph$Credit_History) #Credit_History
data_graph$Property_Area <- as.factor(data_graph$Property_Area) #Property_Area
data_graph$Loan_Status <- factor(data_graph$Loan_Status,labels=c("Rejected", "Approved")) #Loan_Status
summary(data_graph)

#Below are the graphs for all qualitative variables in the set:
#Gender
data_graph %>%
  ggplot(mapping = aes(x = Gender, fill = Gender)) +
  geom_bar(color = "black") +
  labs(title = "Distribution of the Gender Variable",
       x = "Gender",
       y = "Count") +
  scale_fill_manual(values = c("seagreen", "seagreen1")) +
  theme_minimal()

#Married
data_graph %>%
  ggplot(mapping = aes(x = Married, fill = Married)) +
  geom_bar(color = "black") +
  labs(title = "Distribution of the Married Variable",
       x = "Gender",
       y = "Count") +
  scale_fill_manual(values = c("seagreen", "seagreen1")) +
  theme_minimal()

#Dependents
data_graph %>%
  ggplot(mapping = aes(x = Dependents, fill = Dependents)) +
  geom_bar(color = "black") +
  labs(title = "Distribution of the Dependents Variable",
       x = "Dependents",
       y = "Count") +
  scale_fill_manual(values = c("seagreen", "seagreen1", "seagreen3", "seashell3")) +
  theme_minimal()

#Education
data_graph %>%
  ggplot(mapping = aes(x = Education, fill = Education)) +
  geom_bar(color = "black") +
  labs(title = "Distribution of the Education Variable",
       x = "Education",
       y = "Count") +
  scale_fill_manual(values = c("seagreen", "seagreen1"))+
  theme_minimal()

#Self_Employed
data_graph %>%
  ggplot(mapping = aes(x = Self_Employed, fill = Self_Employed)) +
  geom_bar(color = "black") +
  labs(title = "Distribution of the Self_Employed Variable",
       x = "Self_Employed",
       y = "Count") +
  scale_fill_manual(values = c("seagreen", "seagreen1"))+
  theme_minimal()

#Credit_History
data_graph %>%
  ggplot(mapping = aes(x = Credit_History, fill = Credit_History)) +
  geom_bar(color = "black") +
  labs(title = "Distribution of the Credit_History Variable",
       x = "Credit_History",
       y = "Count") +
  scale_fill_manual(values = c("seagreen", "seagreen1")) +
  theme_minimal()

#Property_Area
data_graph %>%
  ggplot(mapping = aes(x = Property_Area, fill = Property_Area)) +
  geom_bar(color = "black") +
  labs(title = "Distribution of the Self_Property Variable",
       x = "Property_Area",
       y = "Count") +
  scale_fill_manual(values = c("seagreen", "seagreen1", "seagreen3")) +
  theme_minimal()

#Loan_Status
data_graph %>%
  ggplot(mapping = aes(x = Loan_Status, fill = Loan_Status)) +
  geom_bar(color = "black") +
  labs(title = "Distribution of the Loan_Status Variable",
       x = "Loan_Status",
       y = "Count") +
  scale_fill_manual(values = c("seagreen", "seagreen1")) +
  theme_minimal()
#As can be seen from the graph, the variable Loan_Status is slightly unbalanced, 
#so that the Approved status is almost 3 times as high as the Rejected status (indicatively).
#(Approved = 242, Rejected = 93)

create_bar_plot <- function(data, variable, title) {
  ggplot(data, aes_string(x = variable, fill = variable)) +
    geom_bar() +
    geom_text(stat='count', aes(label=..count..), vjust=-0.5, color = 'black') +
    ggtitle(title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab(variable) +
    ylab("Count") +
    scale_fill_manual(values = c('seagreen', 'seagreen3')) 
}
plot_LS <- create_bar_plot(dataset, "Loan_Status", "Distribution of Loan Status")
plot_LS

#facet_wrap
#facet_wrap in this context makes visualization more efficient, clean, and easy to interpret. 
#It provides a clear visual comparison of the distributions of qualitative variables and improves the overall experience of data analysis.
data_long <- data_graph %>%
  pivot_longer(cols = c(Gender, Married, Dependents, Education, Self_Employed, Credit_History, Property_Area, Loan_Status), 
               names_to = "Variable", 
               values_to = "Value")
#check the data
head(data_long)
#Create a sufficient color palette
unique_values <- data_long %>% distinct(Value) %>% nrow()
colors <- colorRampPalette(c("seagreen", "seagreen1", "seagreen3", "seashell3"))(unique_values)
#Creating graphs with facet_wrap
data_long_plot = data_long %>%
  ggplot(mapping = aes(x = Value, fill = Value)) +
  geom_bar(color = "black") +
  labs(x = "Value", y = "Count", title = "Distribution of Qualitative Variables") +
  scale_fill_manual(values = colors) +  #Assign dynamically generated color 
  theme_minimal() +
  facet_wrap(~ Variable, scales = "free_x", ncol = 2) +  #ncol = 2 to have two columns in the grid
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#print the graph
print(data_long_plot)
#Explicative note
cat("Note: Some of the graphs shown may not make complete sense due to different measurement scales for the various categories. Please interpret the results with caution.")

#It is therefore useful to factor the Loan_Status variable as well
dataset$Loan_Status <- factor(data_graph$Loan_Status,labels=c("Rejected", "Approved")) #Loan_Status as Factor 
summary(dataset)

#Loan_Status's Statistics
loan_status_stat = dataset %>%
  group_by(Loan_Status) %>%
  summarise(across(where(is.numeric), list(mean=mean, sd=sd), na.rm=TRUE)) #Only the mean and standard deviation are reported as descriptive statistics
loan_status_stat_df <- as.data.frame(loan_status_stat)
print(loan_status_stat_df)
#Main considerations:
#Applicants with approved loans have slightly higher income than rejected applicants, both for the main applicant and co-applicant.
#Loan amounts and durations are similar between the two groups, but approved applicants tend to have a slight difference in average amount and duration.
#Good credit history is a major determinant of loan approval, as indicated by the higher average credit history among approved applicants.


#Numerical Variables's Plots and Histograms 
#Applicant Income refers to the income of the main applicant, Coapplicant Income refers to the income of the second individual participating in the credit application.
#In fact, the Dataset shows us that the latter related to the spouse is considerably different from the Applicant Income and often even equal to 0.

#Applicant Income
#The choice of the binwidth parameter specifically influences the width (size) of the bars in the histogram 
#and has a significant impact on how the income distribution of applicants is displayed. 
#In this case, 200 was selected as the optimal value as it offers a good distribution of the bars.
Applicant_Income_Plot1 = ggplot(dataset, aes(x = Applicant_Income)) +
  geom_histogram(binwidth = 200, fill = 'seagreen', color = 'black') +
  labs(x = "Applicant Income", y = "Frequency") +
  ggtitle("Distribution of Applicants Income") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12))

Applicant_Income_Plot2 <- ggplot(dataset, aes(x = Applicant_Income)) +
  geom_boxplot(fill = "seagreen", color = "black") +  
  labs(x = "Applicant Income", y = "Frequency") + 
  ggtitle("Distribution of Applicants Income") +  
  theme_minimal() +  # Apply minimal theme
  theme(axis.text = element_text(size = 12))  

#Both Histogram and Box-Plot may have limited visibility with the grid.arrange.
grid.arrange(Applicant_Income_Plot1, Applicant_Income_Plot2)
#Graphically, there are values that we can consider as outliers.

#Coapplicant Income
Coapplicant_Income_Plot1 = ggplot(dataset, aes(x = Coapplicant_Income)) +
  geom_histogram(binwidth = 200, fill = 'seagreen', color = 'black') +
  labs(x = "Applicant Income", y = "Frequency") +
  ggtitle("Distribution of Applicants Income") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12))

Coapplicant_Income_Plot2 <- ggplot(dataset, aes(x = Coapplicant_Income)) +
  geom_boxplot(fill = "seagreen", color = "black") +  
  labs(x = "Applicant Income", y = "Frequency") + 
  ggtitle("Distribution of Applicants Income") +  
  theme_minimal() +  # Apply minimal theme
  theme(axis.text = element_text(size = 12))  

grid.arrange(Coapplicant_Income_Plot1, Coapplicant_Income_Plot2)
#It is noticeable of how the distribution is very unbalanced, so this already lets us know that this is a somewhat problematic variable (Presence of Outliers).
#It is advisable to look at the graphs individually because they have scales of measurement ass.

#Loan Amount
Loan_Amount_Plot1 = ggplot(dataset, aes(x = Loan_Amount)) +
  geom_histogram(fill = 'seagreen', color = 'black') +
  labs(x = "Loan Amount", y = "Frequency") +
  ggtitle("Distribution of Loan Amounts") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12))

Loan_Amount_Plot2 = ggplot(dataset, aes(x = Loan_Amount)) +
  geom_boxplot(fill = 'seagreen', color = 'black') +
  labs(x = "Loan Amount", y = "Frequency") +
  ggtitle("Distribution of Loan Amounts") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12))

grid.arrange(Loan_Amount_Plot1, Loan_Amount_Plot2) 
#In this case we already have a better distribution than before, given the lower number of outliers.

#Loan Amount Term
#This variable has a very high number of records equal to 360, 
#it follows that the distribution evaluated graphically is strongly influenced by this constraint.
Loan_Amount_Term_Plot1 = ggplot(dataset, aes(x = Loan_Amount_Term)) + 
  geom_histogram(fill = 'seagreen', color = 'black') +
  labs(x = "Loan Amount Term", y = "Frequency") +
  ggtitle("Distribution of Loan Amounts Term") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12))

Loan_Amount_Term_Plot2 = ggplot(dataset, aes(x = Loan_Amount_Term)) + 
  geom_boxplot(fill = 'seagreen', color = 'black') +
  labs(x = "Loan Amount Term", y = "Frequency") +
  ggtitle("Distribution of Loan Amounts Term") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12))

grid.arrange(Loan_Amount_Term_Plot1, Loan_Amount_Term_Plot2) 
#Already by analyzing the descriptive statistics it could be seen of how this variable took the same value (360.0) for most records. 
#Therefore graphically the representation is very limited.
#A box plot without the box and with only the outliers means that all data are outside the interquartile range (IQR) extended 1.5 times beyond the first and third quartiles.


#######################################################
#Chi-Square Test
#######################################################
library(dplyr)
dataset <- dataset %>%
  mutate(Gender = as.factor(Gender),
         Married = as.factor(Married), 
         Dependents = as.factor(Dependents),
         Education = as.factor(Education),
         Self_Employed = as.factor(Self_Employed),
         Credit_History = as.factor(Credit_History),
         Property_Area = as.factor(Property_Area),
         Loan_Status = as.factor(Loan_Status))
run_chisq_test <- function(variable, outcome) {
  test_result <- chisq.test(table(variable, outcome))
  result_summary <- list(
    statistic = test_result$statistic,
    p_value = test_result$p.value,
    df = test_result$parameter
  )
  return(result_summary)
}
gender_chisq <- run_chisq_test(dataset$Gender, dataset$Loan_Status)
married_chisq <- run_chisq_test(dataset$Married, dataset$Loan_Status)
dependents_chisq <- run_chisq_test(dataset$Dependents, dataset$Loan_Status)
education_chisq <- run_chisq_test(dataset$Education, dataset$Loan_Status)
self_employed_chisq <- run_chisq_test(dataset$Self_Employed, dataset$Loan_Status)
credit_history_chisq <- run_chisq_test(dataset$Credit_History, dataset$Loan_Status)
property_area_chisq <- run_chisq_test(dataset$Property_Area, dataset$Loan_Status)

print("Chi-Square Test Results for Gender")
print(gender_chisq)
print("Chi-Square Test Results for Married")
print(married_chisq)
print("Chi-Square Test Results for Dependents")
print(dependents_chisq)
print("Chi-Square Test Results for Education")
print(education_chisq)
print("Chi-Square Test Results for Self_Employed")
print(self_employed_chisq)
print("Chi-Square Test Results for Credit_History")
print(credit_history_chisq)
print("Chi-Square Test Results for Property_Area")
print(property_area_chisq)
#There are no overly significant associations between Loan Status and the variables Gender, Married, Dependents, Education, and Self-employment.
#In contrast, there are highly significant associations between Loan Status and the variables Credit History and Area of Ownership.


#######################################################
#Operations 
#######################################################
#Dataset restoration operations and final formatting
rm(list=ls())
library(fastDummies)

dataset <- read.csv("loan_data_formatted.csv",sep=";")

cleaned_dataset <- dataset %>% na.omit()
dataset <- cleaned_dataset %>%
  filter(Gender != 0, Married != 0, Self_Employed != 0, Loan_Amount_Term != 0) %>%
  filter(!Loan_ID %in% c("LP001915", "LP002369")) #In this case we eliminated by acting directly on the Loan_IDs since the entire rows were problematic.

str(dataset) #335 observations and 13 variables
names(dataset) 
dim(dataset) #335 rows and 13 columns
summary(dataset)

#Renaming columns
old_labels = colnames(dataset)
new_col_names = c('Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 
                  'Self_Employed', 'Applicant_Income', 'Coapplicant_Income', 
                  'Loan_Amount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status')
colnames(dataset) = new_col_names

#create dummy variables from the char type columns
mod_dataset <- dataset %>%
  dummy_cols(select_columns = c('Gender', 'Married', 'Education', 'Education', 'Self_Employed' ,'Credit_History', 'Loan_Status'), remove_first_dummy = TRUE)
#Being Property_Area a qualitative variable with three possible categories, we create a dummy for each category
mod_dataset$Property_Area_Urban <- ifelse(mod_dataset$Property_Area == "Urban", 1, 0)
mod_dataset$Property_Area_Semiurban <- ifelse(mod_dataset$Property_Area == "Semiurban", 1, 0)
mod_dataset$Property_Area_Rural <- ifelse(mod_dataset$Property_Area == "Rural", 1, 0)

head(mod_dataset)
str(mod_dataset)
dim(mod_dataset)

#For specific reasons we have chosen to remove the following variables from the model, some due to uselessness others dictated by operations that are carried out later in the analysis.
mod_dataset <- subset(mod_dataset, select = -c(Loan_ID, Gender, Married, Education, Self_Employed, Credit_History, Property_Area, Loan_Status))

#View the structure of the data without redundancies
str(mod_dataset)

#Save the modified data frame to a new CSV file
write.csv(mod_dataset, file = "C:/Users/thoma/OneDrive - unibs.it/Statistical Learning/Project/Project_CT_726582/loan_data_formatted2.csv", row.names = FALSE)
