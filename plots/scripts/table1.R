# This file prepares the data for table 1
#
# Derek van Tilborg
# Eindhoven University of Technology
# January 2024


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
  
  FP = confusion[2, ]$Freq
  TP = confusion[4, ]$Freq
  
  PPV = TP / (TP + FP)
  
  return(PPV)
  
}

compute_tpr <- function(y_true, y_hat) {
  # Ensure that y_true and y_hat are factors with the same levels
  y_true <- factor(y_true)
  y_hat <- factor(y_hat, levels = levels(y_true))
  
  confusion = data.frame(table(y_true,y_hat))
  
  TP = confusion[4, ]$Freq
  FN = confusion[3, ]$Freq
  
  TPR  = TP / (TP + FN)
  
  return(TPR)
}


#### load data ####

# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointChemicalModel")

df_3abc <- read_csv('plots/data/df_3abc.csv')

df_3abc = subset(df_3abc, reliability_method %in% c("Embedding dist", "Uncertainty","Unfamiliarity"))
df_3abc$reliability_method = factor(df_3abc$reliability_method, levels = c("Embedding dist", "Uncertainty","Unfamiliarity"))
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


# Ranking correlation of reliability bins
s_table2_ranking_correlation = df_3abc_binned_metrics %>%
  group_by(dataset_name, reliability_method) %>%
  summarize(
    MCSF_cor = cor(as.numeric(as.character(bin)), MCSF, method='kendall'),
    Cats_cos_cor = cor(as.numeric(as.character(bin)), Cats_cos, method='kendall'),
    Tanimoto_scaffold_cor = cor(as.numeric(as.character(bin)), Tanimoto_scaffold_to_train, method='kendall')
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
  mutate(across(where(is.numeric), ~ round(., 3))) 

s_table2_ranking_correlation = s_table2_ranking_correlation %>%
  mutate(across(ends_with("_mean"), 
                ~ paste0(round(., 3), "±", round(s_table2_ranking_correlation[[gsub("_mean$", "_se", cur_column())]], 3)),
                .names = "{.col}")) %>%
  rename_with(~ gsub("_mean", "", .), ends_with("_mean")) %>%
  select(-ends_with("_se"))

s_table2_ranking_correlation


write.csv(s_table2_ranking_correlation, 'plots/tables/table_1.csv', row.names = FALSE)
