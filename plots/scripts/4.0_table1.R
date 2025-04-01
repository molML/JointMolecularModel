# This file prepares the data for table 1
#
# Derek van Tilborg
# Eindhoven University of Technology
# January 2025

library(readr)
library(dplyr)

#### Utils ####

se <- function(x, na.rm = FALSE) {sd(x, na.rm=na.rm) / sqrt(sum(1*(!is.na(x))))}

compute_balanced_accuracy <- function(y_true, y_hat) {
  # Ensure that y_true and y_hat are factors with the same levels
  y_true <- factor(y_true)
  y_hat <- factor(y_hat, levels = levels(y_true))
  
  # Create confusion matrix
  cm <- table(y_true, y_hat)
  
  # Calculate the true positive rate (sensitivity) for each class
  sensitivity_per_class <- diag(cm) / rowSums(cm)
  
  # Calculate the balanced accuracy
  balanced_accuracy <- mean(sensitivity_per_class, na.rm = TRUE)
  
  return(balanced_accuracy)
}

compute_precision <- function(y_true, y_hat) {
  # Ensure that y_true and y_hat are factors with the same levels
  y_true <- factor(y_true)
  y_hat <- factor(y_hat, levels = levels(y_true))
  
  confusion = data.frame(table(y_true,y_hat))
  
  TN = confusion[1, ]$Freq
  FN = confusion[2, ]$Freq
  FP = confusion[3, ]$Freq
  TP = confusion[4, ]$Freq
  
  N = sum(y_true == 0)
  
  PPV = TP / (TP + FP)
  
  return(PPV)
  
}

compute_tpr <- function(y_true, y_hat) {
  # Ensure that y_true and y_hat are factors with the same levels
  y_true <- factor(y_true)
  y_hat <- factor(y_hat, levels = levels(y_true))
  
  confusion = data.frame(table(y_true,y_hat))
  
  TN = confusion[1, ]$Freq
  FN = confusion[2, ]$Freq
  FP = confusion[3, ]$Freq
  TP = confusion[4, ]$Freq
  
  N = sum(y_true == 0)
  
  TPR  = TP / (TP + FN)
  
  return(TPR)
}


#### load data ####

# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointChemicalModel")

df_3binned <- read_csv('plots/data/df_3binned.csv')

df_3binned = subset(df_3binned, reliability_method %in% c("Embedding dist", "Uncertainty","Unfamiliarity"))
df_3binned$reliability_method = factor(df_3binned$reliability_method, levels = c("Unfamiliarity", "Embedding dist", "Uncertainty"))


#### ranking correlation ####

table_1_correlation <- df_3binned %>% 
  group_by(dataset_name, reliability_method) %>%
  summarize(
    MCSF_cor = cor(reliability, MCSF_, method='spearman'),
    Cats_cos_cor = cor(reliability, Cats_cos_, method='spearman'),
    Tanimoto_scaffold_cor = cor(reliability, Tanimoto_scaffold_to_train_, method='spearman')
  ) %>% 
  group_by(reliability_method) %>%
  summarize(
    Tanimoto_scaffold_mean = mean(Tanimoto_scaffold_cor, na.rm = TRUE),
    Tanimoto_scaffold_se = se(Tanimoto_scaffold_cor, na.rm = TRUE),
    MCSF_mean = mean(MCSF_cor, na.rm = TRUE),
    MCSF_se = se(MCSF_cor, na.rm = TRUE),
    Cats_cos_mean = mean(Cats_cos_cor, na.rm = TRUE),
    Cats_cos_se = se(Cats_cos_cor, na.rm = TRUE)
  ) %>% ungroup() %>%
  mutate(across(where(is.numeric), ~ round(., 2))) 


table_1_correlation = table_1_correlation %>%
  mutate(across(ends_with("_mean"), 
                ~ paste0(round(., 2), "±", round(table_1_correlation[[gsub("_mean$", "_se", cur_column())]], 2)),
                .names = "{.col}")) %>%
  rename_with(~ gsub("_mean", "", .), ends_with("_mean")) %>%
  select(-ends_with("_se"))

table_1_correlation

write.csv(table_1_correlation, 'plots/tables/table1.csv', row.names = FALSE)
