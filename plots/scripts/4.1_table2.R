# This file prepares the data for table 2
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
setwd("~/Dropbox/PycharmProjects/JointMolecularModel")

df_3abc <- read_csv('plots/data/df_3binned.csv')

df_3abc$reliability_method = factor(df_3abc$reliability_method, levels = c("Scaffold sim", "Mol core overlap", "Pharmacophore sim", "Embedding dist", "Uncertainty","Unfamiliarity"))
df_3abc$bin = factor(df_3abc$bin)

#### ranking correlation ####

df_3abc_binned_metrics <- df_3abc %>% 
  group_by(dataset_name, reliability_method, bin) %>%
  summarize(
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5)),
    tpr = compute_tpr(y, 1*(y_hat>0.5)),
    precision = compute_precision(y, 1*(y_hat>0.5)),
    MCSF = mean(MCSF_),
    Cats_cos = mean(Cats_cos_),
    Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train_),
    n_bin = n(),
    reliability_bin = mean(reliability)
  ) %>% ungroup() %>% drop_na()


# Check how monotonic the relibability bins are
# monotonic = df_3abc_binned_metrics %>%
#   group_by(dataset_name, reliability_method) %>%
#   summarize(
#     mannkendall = if(n() > 3){MannKendall(balanced_acc)$tau[1]} else {NaN}
#   ) %>% 
#   group_by(reliability_method) %>%
#   summarize(
#     mannkendall.mean = mean(mannkendall, na.rm = TRUE),
#     mannkendall.se = se(mannkendall, na.rm = TRUE)
#   ) %>% ungroup()


# Ranking correlation of reliability bins
table2_ranking_correlation = df_3abc_binned_metrics %>%
  group_by(dataset_name, reliability_method) %>%
  summarize(
    balanced_acc_cor = cor(as.numeric(as.character(bin)), balanced_acc, method='kendall'),
    tpr_cor = cor(as.numeric(as.character(bin)), tpr, method='kendall'),
    precision_cor = cor(as.numeric(as.character(bin)), precision, method='kendall')
  ) %>% 
  group_by(reliability_method) %>%
  summarize(
    balanced_acc_mean = mean(balanced_acc_cor, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc_cor, na.rm = TRUE),
    tpr_mean = mean(tpr_cor, na.rm = TRUE),
    tpr_se = se(tpr_cor, na.rm = TRUE),
    precision_mean = mean(precision_cor, na.rm = TRUE),
    precision_se = se(precision_cor, na.rm = TRUE),
  ) %>% ungroup() %>%
  mutate(across(where(is.numeric), ~ round(., 3))) 

table2_ranking_correlation = table2_ranking_correlation %>%
  mutate(across(ends_with("_mean"), 
                ~ paste0(round(., 2), "±", round(table2_ranking_correlation[[gsub("_mean$", "_se", cur_column())]], 2)),
                .names = "{.col}")) %>%
  rename_with(~ gsub("_mean", "", .), ends_with("_mean")) %>%
  select(-ends_with("_se"))

table2_ranking_correlation

write.csv(table1_ranking_correlation, 'plots/tables/table2.csv', row.names = FALSE)
