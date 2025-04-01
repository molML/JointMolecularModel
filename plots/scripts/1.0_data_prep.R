# This file processes data for all figures in the paper.
#
# Derek van Tilborg
# Eindhoven University of Technology
# January 2024

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

calc_utopia_dist <- function(y_E, confidence, param3 = NULL, maximize_param1 = TRUE, maximize_param2 = TRUE, maximize_param3 = TRUE) {
  # Convert inputs to numeric vectors (if not already)
  y_E <- as.numeric(y_E)
  confidence <- as.numeric(confidence)
  
  # Calculate max and min for normalization
  E_max <- max(y_E)
  E_min <- min(y_E)
  conf_max <- max(confidence)
  conf_min <- min(confidence)
  

    # Normalize bioactivity based on toggle
  if (maximize_param1) {
    norm_bio <- (E_max - y_E) / (E_max - E_min)  # Higher is better
  } else {
    norm_bio <- (y_E - E_min) / (E_max - E_min)  # Lower is better
  }
  
  # Normalize confidence based on toggle
  if (maximize_param2) {
    norm_conf <- (conf_max - confidence) / (conf_max - conf_min)  # Higher is better
  } else {
    norm_conf <- (confidence - conf_min) / (conf_max - conf_min)  # Lower is better
  }
  
  # If param3 is provided, include it in the distance calculation
  if (!is.null(param3)) {
    param3 <- as.numeric(param3)
    param3_max <- max(param3)
    param3_min <- min(param3)
    
    # Normalize param3 based on toggle
    if (maximize_param3) {
      norm_param3 <- (param3_max - param3) / (param3_max - param3_min)  # Higher is better
    } else {
      norm_param3 <- (param3 - param3_min) / (param3_max - param3_min)  # Lower is better
    }
    
    # Compute the Euclidean distance in 3D
    dist_ranking <- sqrt(norm_bio^2 + norm_conf^2 + norm_param3^2)
  } else {
    # Compute the Euclidean distance in 2D
    dist_ranking <- sqrt(norm_bio^2 + norm_conf^2)
  }
  
  return(dist_ranking)
}

#### Main ####

# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointMolecularModel")
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


#### Fig 2 ####

##### a, b, c ####
# Here we describe the relative molecular composition of the data splits per dataset

# summarize the data for every dataset per split
df_2abc = df %>% group_by(split, dataset) %>%
  summarize(Tanimoto_to_train=mean(Tanimoto_to_train),
            Tanimoto_scaffold_to_train=mean(Tanimoto_scaffold_to_train),
            Cats_cos=mean(Cats_cos),
            MCSF=mean(MCSF))

write.csv(df_2abc, 'plots/data/df_2abc.csv', row.names = FALSE)

##### d ####
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


##### e, f, g ####
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
    Tanimoto_to_train = mean(Tanimoto_to_train),
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


##### h ####
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



#### Fig 3 ####

df_2efg$MCSF_ = df_2efg$MCSF
df_2efg$Tanimoto_scaffold_to_train_ = df_2efg$Tanimoto_scaffold_to_train
df_2efg$Tanimoto_to_train_ = df_2efg$Tanimoto_to_train
df_2efg$Cats_cos_ = df_2efg$Cats_cos
df_2efg$y_E_ = df_2efg$y_E

dataset_sizes = data.frame(table(df_2efg$dataset))

# melt dataframe
df_3binned <- df_2efg %>%
  pivot_longer(
    cols = c("mean_z_dist", "ood_score", 'y_unc', 'MCSF', 'Tanimoto_scaffold_to_train', 'Cats_cos', 'y_E', 'Tanimoto_to_train'), # Columns to melt.
    names_to = "reliability_method",    # Name of the new column for method names
    values_to = "reliability"           # Name of the new column for values
  ) %>%
  select(split, dataset, dataset_name, y_hat, smiles, y, y_E_, MCSF_, Tanimoto_scaffold_to_train_, Cats_cos_, Tanimoto_to_train_, split_balanced_acc, reliability_method, reliability)

# rename
df_3binned$reliability_method = gsub('mean_z_dist', 'Embedding dist', df_3binned$reliability_method)
df_3binned$reliability_method = gsub('ood_score', 'Unfamiliarity', df_3binned$reliability_method)
df_3binned$reliability_method = gsub('y_unc', 'Uncertainty', df_3binned$reliability_method)
df_3binned$reliability_method = gsub('MCSF', 'Mol core overlap', df_3binned$reliability_method)
df_3binned$reliability_method = gsub('Tanimoto_scaffold_to_train', 'Scaffold sim', df_3binned$reliability_method)
df_3binned$reliability_method = gsub('Cats_cos', 'Pharmacophore sim', df_3binned$reliability_method)
df_3binned$reliability_method = gsub('Tanimoto_to_train', 'Substructure sim', df_3binned$reliability_method)
df_3binned$reliability_method = gsub('y_E', 'Expected value', df_3binned$reliability_method)

# Invert the reliability metrics so it becomes a 'confidence score'. This is to make the calibration curves and bin correlations more interpretable
df_3binned$reliability[df_3binned$reliability_method == 'Embedding dist'] = -1 * df_3binned$reliability[df_3binned$reliability_method == 'Embedding dist']
df_3binned$reliability[df_3binned$reliability_method == 'Unfamiliarity'] = -1 * df_3binned$reliability[df_3binned$reliability_method == 'Unfamiliarity']
df_3binned$reliability[df_3binned$reliability_method == 'Uncertainty'] = -1 * df_3binned$reliability[df_3binned$reliability_method == 'Uncertainty']

# Bin reliability values per dataset and method
df_3binned <- df_3binned %>% group_by(dataset_name, reliability_method) %>% # 
  mutate(bin = factor(ntile(reliability, 8))
  ) %>% ungroup()

# Invert the reliability metrics back to their original values.
df_3binned$reliability[df_3binned$reliability_method == 'Embedding dist'] = -1 * df_3binned$reliability[df_3binned$reliability_method == 'Embedding dist']
df_3binned$reliability[df_3binned$reliability_method == 'Unfamiliarity'] = -1 * df_3binned$reliability[df_3binned$reliability_method == 'Unfamiliarity']
df_3binned$reliability[df_3binned$reliability_method == 'Uncertainty'] = -1 * df_3binned$reliability[df_3binned$reliability_method == 'Uncertainty']

# Data used for table 1&2
write.csv(df_3binned, 'plots/data/df_3binned.csv', row.names = FALSE)


df_3 = df_2efg %>% group_by(dataset) %>% # 
  mutate(utopia_dist_E = calc_utopia_dist(y_E, y_E, maximize_param2=TRUE),
         utopia_dist_min_unc = calc_utopia_dist(y_unc, y_unc, maximize_param1=FALSE, maximize_param2=FALSE),
         utopia_dist_min_ood = calc_utopia_dist(ood_score, ood_score, maximize_param1=FALSE, maximize_param2=FALSE),
         
         utopia_dist_tanimoto = calc_utopia_dist(Tanimoto_to_train, Tanimoto_to_train, maximize_param2=TRUE),
         utopia_dist_scaffold_tanimoto = calc_utopia_dist(Tanimoto_scaffold_to_train, Tanimoto_scaffold_to_train, maximize_param2=TRUE),
         utopia_dist_mcsf = calc_utopia_dist(MCSF, MCSF, maximize_param2=TRUE),
         utopia_dist_cats = calc_utopia_dist(Cats_cos, Cats_cos, maximize_param2=TRUE),
         
         utopia_dist_E_min_unc = calc_utopia_dist(y_E, y_unc, maximize_param2=FALSE),
         utopia_dist_E_max_unc = calc_utopia_dist(y_E, y_unc, maximize_param2=TRUE),
         utopia_dist_E_min_ood = calc_utopia_dist(y_E, ood_score, maximize_param2=FALSE),
         utopia_dist_E_max_ood = calc_utopia_dist(y_E, ood_score, maximize_param2=TRUE),
         
         utopia_dist_E_min_unc_max_ood = calc_utopia_dist(y_E, y_unc, ood_score, maximize_param2=FALSE, maximize_param3=TRUE),
         utopia_dist_E_min_unc_min_ood = calc_utopia_dist(y_E, y_unc, ood_score, maximize_param2=FALSE, maximize_param3=FALSE),
         
         utopia_dist_E_min_tanimoto = calc_utopia_dist(y_E, Tanimoto_to_train, maximize_param2=FALSE),
         utopia_dist_E_min_scaffold_tanimoto = calc_utopia_dist(y_E, Tanimoto_scaffold_to_train, maximize_param2=FALSE),
         utopia_dist_E_min_mcsf = calc_utopia_dist(y_E, MCSF, maximize_param2=FALSE),
         utopia_dist_E_min_cats = calc_utopia_dist(y_E, Cats_cos, maximize_param2=FALSE),
         
         utopia_dist_E_min_unc_min_tanimoto = calc_utopia_dist(y_E, y_unc, Tanimoto_to_train, maximize_param2=FALSE, maximize_param3=FALSE),
         utopia_dist_E_min_unc_min_scaffold_tanimoto = calc_utopia_dist(y_E, y_unc, Tanimoto_scaffold_to_train, maximize_param2=FALSE, maximize_param3=FALSE),
         utopia_dist_E_min_unc_min_mcsf = calc_utopia_dist(y_E, y_unc, MCSF, maximize_param2=FALSE, maximize_param3=FALSE),
         utopia_dist_E_min_unc_min_cats = calc_utopia_dist(y_E, y_unc, Cats_cos, maximize_param2=FALSE, maximize_param3=FALSE),
         
         utopia_dist_E_min_ood_min_tanimoto = calc_utopia_dist(y_E, ood_score, Tanimoto_to_train, maximize_param2=FALSE, maximize_param3=FALSE),
         utopia_dist_E_min_ood_min_scaffold_tanimoto = calc_utopia_dist(y_E, ood_score, Tanimoto_scaffold_to_train, maximize_param2=FALSE, maximize_param3=FALSE),
         utopia_dist_E_min_ood_min_mcsf = calc_utopia_dist(y_E, ood_score, MCSF, maximize_param2=FALSE, maximize_param3=FALSE),
         utopia_dist_E_min_ood_min_cats = calc_utopia_dist(y_E, ood_score, Cats_cos, maximize_param2=FALSE, maximize_param3=FALSE),
         
         utopia_dist_min_ood_min_tanimoto = calc_utopia_dist(ood_score, Tanimoto_to_train, maximize_param1=FALSE, maximize_param2=FALSE),
         utopia_dist_min_ood_min_scaffold_tanimoto = calc_utopia_dist(ood_score, Tanimoto_scaffold_to_train, maximize_param1=FALSE, maximize_param2=FALSE),
         utopia_dist_min_ood_min_mcsf = calc_utopia_dist(ood_score, MCSF, maximize_param1=FALSE, maximize_param2=FALSE),
         utopia_dist_min_ood_min_cats = calc_utopia_dist(ood_score, Cats_cos, maximize_param1=FALSE, maximize_param2=FALSE)
         ) %>% 
  ungroup()

# melt dataframe
df_3 <- df_3 %>%
  pivot_longer(
    cols = starts_with("utopia_dist"), # Columns to melt.
    names_to = "ranking_method",    # Name of the new column for method names
    values_to = "utopia_dist"           # Name of the new column for values
  ) %>%
  select(split, dataset, dataset_name, y_hat, smiles, y, y_E_, y_unc, ood_score, MCSF_, Tanimoto_scaffold_to_train_, Cats_cos_, Tanimoto_to_train_, split_balanced_acc, ranking_method, utopia_dist)


write.csv(df_3, 'plots/data/df_3.csv', row.names = FALSE)
