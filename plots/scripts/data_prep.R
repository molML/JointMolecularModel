# This file processes data for all figures in the paper.
#
# Derek van Tilborg
# Eindhoven University of Technology
# Januari 2024

#### libraries ####

library(readr)
library(ggplot2)
library(dplyr)
library(cowplot)
library(ggridges)
library(viridis)
library(hrbrthemes)
library(data.table)
library(patchwork)
library(caret)
library(stringr)
library(ggrepel)
library(factoextra)
library(tidyr)

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

#### Main ####

# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointChemicalModel")
df <- read_csv("results/processed/all_results_processed.csv")

df$split = gsub('train', 'Train', gsub('test', 'Test', gsub('ood', 'OOD', df$split)))
df$split = factor(df$split, levels = c('Train', 'Test', 'OOD'))
df$descriptor = toupper(df$descriptor)
df$descriptor = factor(df$descriptor, levels = c('ECFP', 'CATS', 'SMILES'))
df$model_type = toupper(df$model_type)
df$model_type = factor(df$model_type, levels = c('MLP', 'RF', 'JMM', 'AE'))
df$method = paste0(df$descriptor, '_', df$model_type)
df = subset(df, method != "CATS_MLP")
df$method = factor(df$method, levels = c("CATS_RF", "ECFP_RF", "ECFP_MLP", "SMILES_MLP", "SMILES_JMM", 'SMILES_AE'))
df$ood_score = log(df$reconstruction_loss)
df$y_hat = factor(df$y_hat)
df$y = factor(df$y)
df$correct_pred = 1*(df$y_hat == df$y)


#### Fig 2abc ####
# Here we describe the relative molecular composition of the data splits per dataset

# summarize the data for every dataset per split
df_2abc = df %>% group_by(split, dataset) %>%
  summarize(Tanimoto_to_train=mean(Tanimoto_to_train),
            Tanimoto_scaffold_to_train=mean(Tanimoto_scaffold_to_train),
            Cats_cos=mean(Cats_cos),
            MCSF=mean(MCSF))

write.csv(df_2abc, 'plots/data/df_2abc.csv', row.names = FALSE)

#### Fig 2d ####
# Here we describe model predictive performance

# Summarize the data so it has metrics per dataset
df_2d = subset(df, method != "SMILES_AE" & split != 'Train')
df_2d <- df_2d %>%
  group_by(split, dataset, method) %>%
  summarize(
    accuracy = mean(split_acc),
    TPR = mean(split_TPR),
    TNR = mean(split_TNR),
    balanced_accuracy = mean(split_balanced_acc),
    uncertainty = mean(y_unc)
  )

write.csv(df_2d, 'plots/data/df_2d.csv', row.names = FALSE)


#### Fig 2efg ####
# Here we describe the OOD score globally

# summarize the data per sample (spanning all datasets) per split
df_2efg = subset(df, method == "SMILES_JMM" & split != 'Train') %>%  #
  group_by(split, dataset, smiles) %>%
  summarize(
    ood_score = mean(ood_score),
    correct_pred = mean(correct_pred),
    sdc_ad = mean(sdc_ad),
    mean_z_dist = mean(mean_z_dist),
    y_hat = mean(as.numeric(as.character(y_hat))),
    y = mean(as.numeric(as.character(y))),
    y_E = mean(y_E),
    y_unc = mean(y_unc),
    split_balanced_acc = mean(split_balanced_acc),
    Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train),
    MCSF = mean(MCSF),
    Cats_cos = mean(Cats_cos),
    bertz = mean(bertz),
    bottcher = mean(bottcher),
    molecule_entropy = mean(molecule_entropy),
    motifs = mean(motifs),
    n_smiles_tokens = mean(n_smiles_tokens),
    n_rings = mean(n_rings),
    n_smiles_branches = mean(n_smiles_branches),
    mol_weight = mean(mol_weight),
    smiles_entropy = mean(smiles_entropy)) %>% 
  ungroup() %>% group_by(dataset) %>%
  mutate(quartile_ood = factor(ntile(ood_score, 8))
  ) %>% ungroup()

df_2efg$y = factor(df_2efg$y)
df_2efg$y_hat_binary = factor(1*(df_2efg$y_hat > 0.5))

# Change the dataset names to their target name
# LitPCBA and MoleculeACE (CHEMBL3979_EC50) both have a PPAR gamma dataset, I will label them with a reference in post processing
target_names = data.frame(id = c("PPARG", "Ames_mutagenicity", "ESR1_ant", "TP53", "CHEMBL1871_Ki","CHEMBL218_EC50","CHEMBL244_Ki","CHEMBL236_Ki","CHEMBL234_Ki","CHEMBL219_Ki","CHEMBL238_Ki","CHEMBL4203_Ki","CHEMBL2047_EC50","CHEMBL4616_EC50","CHEMBL2034_Ki","CHEMBL262_Ki","CHEMBL231_Ki","CHEMBL264_Ki","CHEMBL2835_Ki","CHEMBL2971_Ki","CHEMBL237_EC50","CHEMBL237_Ki","CHEMBL233_Ki","CHEMBL4792_Ki","CHEMBL239_EC50","CHEMBL3979_EC50","CHEMBL235_EC50","CHEMBL4005_Ki","CHEMBL2147_Ki","CHEMBL214_Ki","CHEMBL228_Ki","CHEMBL287_Ki","CHEMBL204_Ki","CHEMBL1862_Ki"),
                          name = c("PPARyl", "Ames", "ESR1", "TP53", "AR","CB1","FX","DOR","D3R","D4R","DAT","CLK4","FXR","GHSR","GR","GSK3","HRH1","HRH3","JAK1","JAK2","KOR (a)","KOR (i)","MOR","OX2R","PPARa","PPARym","PPARd","PIK3CA","PIM1","5-HT1A","SERT","SOR","Thrombin","ABL1"))
df_2efg$dataset_name = target_names$name[match(df_2efg$dataset, target_names$id)]

write.csv(df_2efg, 'plots/data/df_2efg.csv', row.names = FALSE)


#### 2h ####
# Here we describe the relationship between OOD score and distance to the train data

# Compute the balanced accuracy per bin. Compute the mean and se over datasets
df_2h = df_2efg %>% 
  group_by(dataset, split, quartile_ood) %>%
  summarise(
    y_unc = mean(y_unc),
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    mean_z_dist = mean(mean_z_dist),
    Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train),
    Cats_cos = mean(Cats_cos),
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5))
  ) %>% ungroup() %>% 
  group_by(split, quartile_ood) %>%
  summarise(
    y_unc_mean = mean(y_unc, na.rm = TRUE),
    y_unc_se = sd(y_unc, na.rm = TRUE),
    ood_score_mean = mean(ood_score, na.rm = TRUE),
    ood_score_se = sd(ood_score, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    mean_z_dist_mean = mean(mean_z_dist, na.rm = TRUE),
    mean_z_dist_se = se(mean_z_dist, na.rm = TRUE),
    Tanimoto_scaffold_to_train_mean = mean(Tanimoto_scaffold_to_train, na.rm = TRUE),
    Tanimoto_scaffold_to_train_se = se(Tanimoto_scaffold_to_train, na.rm = TRUE),
    Cats_cos_mean = mean(Cats_cos, na.rm = TRUE),
    Cats_cos_se = se(Cats_cos, na.rm = TRUE),
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE)
  ) %>% ungroup()


write.csv(df_2h, 'plots/data/df_2h.csv', row.names = FALSE)



#### Fig 3abc ####

df_2efg$MCSF_ = df_2efg$MCSF
df_2efg$Tanimoto_scaffold_to_train_ = df_2efg$Tanimoto_scaffold_to_train
df_2efg$Cats_cos_ = df_2efg$Cats_cos

# melt dataframe
df_3abc <- df_2efg %>%
  pivot_longer(
    cols = c("mean_z_dist", "ood_score", 'y_unc', 'MCSF', 'Tanimoto_scaffold_to_train', 'Cats_cos'), # Columns to melt. 'Cats_cos', 'bertz', 'Tanimoto_scaffold_to_train', "sdc_ad", 'MCSF'
    names_to = "reliability_method",          # Name of the new column for method names
    values_to = "reliability"           # Name of the new column for values
  ) %>%
  select(split, dataset_name, y_hat, smiles, y, y_E, MCSF_, Tanimoto_scaffold_to_train_, Cats_cos_, split_balanced_acc, reliability_method, reliability)

# rename
df_3abc$reliability_method = gsub('mean_z_dist', 'Embedding dist', df_3abc$reliability_method)
df_3abc$reliability_method = gsub('ood_score', 'Unfamiliarity', df_3abc$reliability_method)
df_3abc$reliability_method = gsub('y_unc', 'Uncertainty', df_3abc$reliability_method)
df_3abc$reliability_method = gsub('MCSF', 'Mol core overlap', df_3abc$reliability_method)
df_3abc$reliability_method = gsub('Tanimoto_scaffold_to_train', 'Scaffold sim', df_3abc$reliability_method)
df_3abc$reliability_method = gsub('Cats_cos', 'Pharmacophore sim', df_3abc$reliability_method)

# Invert the reliability metrics so it becomes a 'confidence score'. This is to make the calibration curves and bin correlations more interpretable
df_3abc$reliability[df_3abc$reliability_method == 'Embedding dist'] = -1 * df_3abc$reliability[df_3abc$reliability_method == 'Embedding dist']
df_3abc$reliability[df_3abc$reliability_method == 'Unfamiliarity'] = -1 * df_3abc$reliability[df_3abc$reliability_method == 'Unfamiliarity']
df_3abc$reliability[df_3abc$reliability_method == 'Uncertainty'] = -1 * df_3abc$reliability[df_3abc$reliability_method == 'Uncertainty']

# Bin reliability values per dataset and method
df_3abc <- df_3abc %>% group_by(dataset_name, reliability_method) %>% # 
  mutate(bin = factor(ntile(reliability, 10))
  ) %>% ungroup()

# Invert the reliability metrics back to their original values.
df_3abc$reliability[df_3abc$reliability_method == 'Embedding dist'] = -1 * df_3abc$reliability[df_3abc$reliability_method == 'Embedding dist']
df_3abc$reliability[df_3abc$reliability_method == 'Unfamiliarity'] = -1 * df_3abc$reliability[df_3abc$reliability_method == 'Unfamiliarity']
df_3abc$reliability[df_3abc$reliability_method == 'Uncertainty'] = -1 * df_3abc$reliability[df_3abc$reliability_method == 'Uncertainty']


write.csv(df_3abc, 'plots/data/df_3abc.csv', row.names = FALSE)

