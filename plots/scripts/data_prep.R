# This file processes data for all figures in the paper.
#
# Derek van Tilborg
# Eindhoven University of Technology
# Januari 2024


# loading some libraries
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



#### Calibration ####



jvae_reconstruction$MCSF_ = jvae_reconstruction$MCSF
jvae_reconstruction$Tanimoto_scaffold_to_train_ = jvae_reconstruction$Tanimoto_scaffold_to_train
jvae_reconstruction$Cats_cos_ = jvae_reconstruction$Cats_cos

jvae_calibration <- jvae_reconstruction %>%
  pivot_longer(
    cols = c("mean_z_dist", "ood_score", 'y_unc', 'MCSF', 'Tanimoto_scaffold_to_train', 'Cats_cos'), # Columns to melt. 'Cats_cos', 'bertz', 'Tanimoto_scaffold_to_train', "sdc_ad", 'MCSF'
    names_to = "reliability_method",          # Name of the new column for method names
    values_to = "reliability"           # Name of the new column for values
  ) %>%
  select(split, dataset_name, y_hat, y, y_E, MCSF_, Tanimoto_scaffold_to_train_, Cats_cos_, split_balanced_acc, reliability_method, reliability)

jvae_calibration$reliability[jvae_calibration$reliability_method == 'MCSF'] = 1 - jvae_calibration$reliability[jvae_calibration$reliability_method == 'MCSF']
jvae_calibration$reliability[jvae_calibration$reliability_method == 'Tanimoto_scaffold_to_train'] = 1 - jvae_calibration$reliability[jvae_calibration$reliability_method == 'Tanimoto_scaffold_to_train']
jvae_calibration$reliability[jvae_calibration$reliability_method == 'Cats_cos'] = 1 - jvae_calibration$reliability[jvae_calibration$reliability_method == 'Cats_cos']

# Bin reliability values per dataset and method
binned_data <- jvae_calibration %>% group_by(dataset_name, reliability_method) %>% # 
  mutate(bin = factor(ntile(reliability, 10))
  ) %>% ungroup()

binned_data_metrics <- binned_data %>% 
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


# ECE

library("Kendall")
monotonic = binned_data_metrics %>%
  group_by(dataset_name, reliability_method) %>%
  summarize(
    # mannkendall = MannKendall(balanced_acc)$tau[1]
    mannkendall = if(n() > 3){MannKendall(balanced_acc)$tau[1]} else {NaN}
    
  ) %>% 
  group_by(reliability_method) %>%
  summarize(
    mannkendall.mean = mean(mannkendall, na.rm = TRUE),
    mannkendall.se = se(mannkendall, na.rm = TRUE)
  ) %>% ungroup()



# Ranking correlation

ranking_correlation = binned_data_metrics %>%
  group_by(dataset_name, reliability_method) %>%
  summarize(
    balanced_acc_cor = cor(as.numeric(as.character(bin)), balanced_acc, method='kendall'),
    tpr_cor = cor(as.numeric(as.character(bin)), tpr, method='kendall'),
    precision_cor = cor(as.numeric(as.character(bin)), precision, method='kendall'),
    MCSF_cor = cor(as.numeric(as.character(bin)), MCSF, method='kendall'),
    Cats_cos_cor = cor(as.numeric(as.character(bin)), Cats_cos, method='kendall'),
    Tanimoto_scaffold_cor = cor(as.numeric(as.character(bin)), Tanimoto_scaffold_to_train, method='kendall')
  ) %>% 
  group_by(reliability_method) %>%
  summarize(
    balanced_acc_mean = mean(balanced_acc_cor, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc_cor, na.rm = TRUE),
    tpr_mean = mean(tpr_cor, na.rm = TRUE),
    tpr_se = se(tpr_cor, na.rm = TRUE),
    precision_mean = mean(precision_cor, na.rm = TRUE),
    precision_se = se(precision_cor, na.rm = TRUE),
    MCSF_mean = mean(MCSF_cor, na.rm = TRUE),
    MCSF_se = se(MCSF_cor, na.rm = TRUE),
    Cats_cos_mean = mean(Cats_cos_cor, na.rm = TRUE),
    Cats_cos_se = se(Cats_cos_cor, na.rm = TRUE),
    Tanimoto_scaffold_mean = mean(Tanimoto_scaffold_cor, na.rm = TRUE),
    Tanimoto_scaffold_se = se(Tanimoto_scaffold_cor, na.rm = TRUE)
  ) %>% ungroup()

ranking_correlation


# plots

calibration_summary <- binned_data_metrics %>% 
  group_by(reliability_method, bin) %>%
  summarise(
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE),
    tpr_mean = mean(tpr, na.rm = TRUE),
    tpr_se = se(tpr, na.rm = TRUE),
    precision_mean = mean(precision, na.rm = TRUE),
    precision_se = se(precision, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    Cats_cos_mean = mean(Cats_cos, na.rm = TRUE),
    Cats_cos_se = se(Cats_cos, na.rm = TRUE),
    Tanimoto_scaffold_to_train_mean = mean(Tanimoto_scaffold_to_train, na.rm = TRUE),
    Tanimoto_scaffold_to_train_se = se(Tanimoto_scaffold_to_train),
  ) %>% ungroup()

calibration_summary$bin = as.numeric(as.character(calibration_summary$bin)) - 1

calibration_summary$reliability_method = gsub('Cats_cos', 'Pharmacophore sim', calibration_summary$reliability_method)
calibration_summary$reliability_method = gsub('MCSF', 'Mol. core overlap', calibration_summary$reliability_method)
calibration_summary$reliability_method = gsub('Tanimoto_scaffold_to_train', 'Scaffold sim', calibration_summary$reliability_method)
calibration_summary$reliability_method = gsub('mean_z_dist', 'Embedding distance', calibration_summary$reliability_method)
calibration_summary$reliability_method = gsub('ood_score', 'Unfamiliarity', calibration_summary$reliability_method)
calibration_summary$reliability_method = gsub('y_unc', 'Uncertainty', calibration_summary$reliability_method)

calibration_summary$reliability_method = factor(calibration_summary$reliability_method, 
                                                levels = c("Pharmacophore sim", 
                                                           "Scaffold sim",
                                                           "Mol. core overlap", 
                                                           "Embedding distance", 
                                                           "Uncertainty",
                                                           "Unfamiliarity"))

fig3a = ggplot(calibration_summary, aes(y=balanced_acc_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = balanced_acc_mean - balanced_acc_se, ymax = balanced_acc_mean + balanced_acc_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  coord_cartesian(ylim=c(0.40, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Balanced accuracy', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 

fig3b = ggplot(calibration_summary, aes(y=tpr_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = tpr_mean - tpr_se, ymax = tpr_mean + tpr_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  coord_cartesian(ylim=c(0.4, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Hit rate', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 

fig3c = ggplot(calibration_summary, aes(y=precision_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = precision_mean - precision_se, ymax = precision_mean + precision_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  coord_cartesian(ylim=c(0.40, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Precision', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'right',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 


fig3 = plot_grid(fig3a, fig3b, fig3c, ncol=3, labels = c('a', 'b', 'c'), rel_widths = c(1,1,1.8), label_size = 10)
fig3


# save to pdf
pdf('plots/fig3.pdf', width = 180/25.4, height = 45/25.4)
print(fig3)
dev.off()


sfig4a = ggplot(calibration_summary, aes(y=MCSF_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = MCSF_mean - MCSF_se, ymax = MCSF_mean + MCSF_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  # coord_cartesian(ylim=c(0.40, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Mol core overlap', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 

sfig4b = ggplot(calibration_summary, aes(y=Cats_cos_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = Cats_cos_mean - Cats_cos_se, ymax = Cats_cos_mean + Cats_cos_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  # coord_cartesian(ylim=c(0.4, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Pharmacophore sim', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 

sfig4c = ggplot(calibration_summary, aes(y=Tanimoto_scaffold_to_train_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = Tanimoto_scaffold_to_train_mean - Tanimoto_scaffold_to_train_se, ymax = Tanimoto_scaffold_to_train_mean + Tanimoto_scaffold_to_train_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  # coord_cartesian(ylim=c(0.40, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Scaffold sim', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'right',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 


sfig4 = plot_grid(sfig4a, sfig4b, sfig4c, ncol=3, labels = c('a', 'b', 'c'), rel_widths = c(1,1,1.8), label_size = 10)
sfig4
# ggplot(calibration_summary, aes(y=MCSF_mean, x=bin, color=reliability_method, fill=reliability_method))+
#   geom_line(size=0.35)+
#   geom_ribbon(aes(ymin = MCSF_mean - MCSF_se, ymax = MCSF_mean + MCSF_se), size=0, alpha=0.1) +
#   # coord_cartesian(ylim=c(0.50, 1), xlim=c(0, 9))+
#   scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
#   labs(y='MCSF', x='bins') +
#   default_theme + theme(legend.position = 'right',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 
# 
# 
# ggplot(calibration_summary, aes(y=Cats_cos_mean, x=bin, color=reliability_method, fill=reliability_method))+
#   geom_line(size=0.35)+
#   geom_ribbon(aes(ymin = Cats_cos_mean - Cats_cos_se, ymax = Cats_cos_mean + Cats_cos_se), size=0, alpha=0.1) +
#   # coord_cartesian(ylim=c(0.50, 1), xlim=c(0, 9))+
#   scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
#   labs(y='Cats_cos', x='bins') +
#   default_theme + theme(legend.position = 'right',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 
# 
# 
# ggplot(calibration_summary, aes(y=Tanimoto_scaffold_to_train_mean, x=bin, color=reliability_method, fill=reliability_method))+
#   geom_line(size=0.35)+
#   geom_ribbon(aes(ymin = Tanimoto_scaffold_to_train_mean - Tanimoto_scaffold_to_train_se, ymax = Tanimoto_scaffold_to_train_mean + Tanimoto_scaffold_to_train_se), size=0, alpha=0.1) +
#   # coord_cartesian(ylim=c(0.50, 1), xlim=c(0, 9))+
#   scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
#   labs(y='Tanimoto_scaffold_to_train', x='bins') +
#   default_theme + theme(legend.position = 'right',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 

# write.csv(jvae_calibration, 'jvae_calibration.csv')

# bin predictions and bin reliability

reliability_curves <- jvae_calibration %>%
  group_by(dataset_name, reliability_method, confidence_bin) %>%
  summarize(
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5))
    # balanced_acc = compute_tpr(y, 1*(y_hat>0.5))
  ) %>% ungroup() %>% drop_na() %>% 
  group_by(reliability_method, confidence_bin) %>%
  summarise(
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE)
  ) %>% ungroup()


# ECE

# jvae_calibration$reliability_bin

calibration_data <- jvae_calibration %>%
  group_by(dataset_name, reliability_method, confidence_bin) %>%
  summarize(
    mean_conf = mean(confidence),
    mean_acc = mean(compute_balanced_accuracy(y, 1*(y_hat>0.5))),
    # mean_acc = mean(compute_tpr(y, 1*(y_hat>0.5))),
    count = n()
  ) %>% ungroup() %>% drop_na()


# Step 5: Calculate ECE for each method and dataset
ece_data <- calibration_data %>%
  group_by(dataset_name, reliability_method) %>%
  summarize(
    ece = sum((count / sum(count)) * abs(mean_acc - mean_conf))
  ) %>% ungroup()

# order and convert bins to numerics
ece_data$reliability_method = factor(ece_data$reliability_method, levels = c('y_unc', 'mean_z_dist', 'ood_score', 'MCSF'))
reliability_curves$reliability_method = factor(reliability_curves$reliability_method, levels = c('y_unc', 'mean_z_dist', 'ood_score', 'MCSF'))

reliability_curves$confidence_bin = as.numeric(as.character(reliability_curves$confidence_bin)) - 1

ggplot(reliability_curves, aes(y=balanced_acc_mean, x=confidence_bin, color=reliability_method, fill=reliability_method))+
  geom_line(size=0.35)+
  geom_ribbon(aes(ymin = balanced_acc_mean - balanced_acc_se, ymax = balanced_acc_mean + balanced_acc_se), size=0, alpha=0.1) +
  coord_cartesian(ylim=c(0.50, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  geom_abline(slope=1/10/2, intercept = 0.5, linetype = "dashed", color = "black") +
  labs(y='Balanced accuracy', x='bins') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(ece_data, aes(x = reliability_method, y = ece, fill = reliability_method)) +
  geom_jitter(aes(fill=reliability_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_x_discrete(labels = c("uncertainty", "z distance", "unfamiliarity", "MCSF")) +
  labs(x = '', y = 'ECE') +
  # scale_fill_manual(values = c('#97a4ab','#efc57b')) +
  # scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(plot.margin = unit(c(0.2, 0.2, -0.075, 0.2), "cm"), legend.position = 'right') 






# # Step 6: Summarize ECE across datasets
ece_summary <- ece_data %>%
  group_by(reliability_method) %>%
  summarize(
    mean_ece = mean(ece),
    ece_sd = sd(ece),
    .groups = 'drop'
  )


subset(ece_data, reliability_method == 'ood_score')$ece
subset(ece_data, reliability_method == 'y_unc')$ece

# %>% 
# group_by(reliability_method, y_E_binned) %>%
# summarise(
#   balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
#   balanced_acc_se = se(balanced_acc, na.rm = TRUE)
# ) %>% ungroup()



# x = predicted probability per bin
# y = observed probability per bin







# Compute the balanced accuracy per bin. Compute the mean and se over datasets
jvae_reconstruction_unc_binned = jvae_reconstruction %>% 
  group_by(dataset, split, quartile_unc) %>%
  summarise(
    y_unc = mean(y_unc),
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train),
    Cats_cos = mean(Cats_cos),
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5))
  ) %>% ungroup() %>% 
  group_by(split, quartile_unc) %>%
  summarise(
    y_unc_mean = mean(y_unc, na.rm = TRUE),
    y_unc_se = sd(y_unc, na.rm = TRUE),
    ood_score_mean = mean(ood_score, na.rm = TRUE),
    ood_score_se = sd(ood_score, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    Tanimoto_scaffold_to_train_mean = mean(Tanimoto_scaffold_to_train, na.rm = TRUE),
    Tanimoto_scaffold_to_train_se = se(Tanimoto_scaffold_to_train, na.rm = TRUE),
    Cats_cos_mean = mean(Cats_cos, na.rm = TRUE),
    Cats_cos_se = se(Cats_cos, na.rm = TRUE),
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE)
  ) %>% ungroup()



ggplot(jvae_reconstruction_ood_binned, aes(y=balanced_acc_mean, x=quartile_ood, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  coord_cartesian(ylim=c(0.5, 0.9))+
  labs(y='Balanced accuracy', x='OOD score bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(jvae_reconstruction_unc_binned, aes(y=balanced_acc_mean, x=quartile_unc, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  coord_cartesian(ylim=c(0.5, 0.9))+
  labs(y='Balanced accuracy', x='Uncertainty bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(jvae_reconstruction_unc_binned, aes(y=ood_score_mean, x=quartile_unc, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = ood_score_mean-ood_score_se, ymax = ood_score_mean+ood_score_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  coord_cartesian(ylim=c(0.3, 1.1))+
  labs(y='OOD score', x='Uncertainty bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(jvae_reconstruction, aes(y=ood_score, x=log(y_unc))) +
  labs(y='OOD score', x='log(uncertainty)') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  # scale_y_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))



ggplot(jvae_reconstruction_unc_binned, aes(y=MCSF_mean, x=quartile_unc, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = MCSF_mean-MCSF_se, ymax = MCSF_mean+MCSF_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  # coord_cartesian(ylim=c(0.5, 0.9))+
  # labs(y='Balanced accuracy', x='OOD score bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(jvae_reconstruction_unc_binned, aes(y=Tanimoto_scaffold_to_train_mean, x=quartile_unc, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = Tanimoto_scaffold_to_train_mean-Tanimoto_scaffold_to_train_se, ymax = Tanimoto_scaffold_to_train_mean+Tanimoto_scaffold_to_train_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  # coord_cartesian(ylim=c(0.5, 0.9))+
  # labs(y='Balanced accuracy', x='Uncertainty bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(jvae_reconstruction_unc_binned, aes(y=Cats_cos_mean, x=quartile_unc, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = Cats_cos_mean-Cats_cos_se, ymax = Cats_cos_mean+Cats_cos_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  # coord_cartesian(ylim=c(0.3, 1.1))+
  # labs(y='OOD score', x='Uncertainty bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 






# 
# ggplot(jvae_reconstruction, aes(y=ood_score, x=quartile_unc, color=split, fill=split))+
#   # geom_point(size=0.35)+
#   geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
#                               outlier.shape=NA, varwidth = FALSE, lwd=0.3, fatten=0.75) +
#   # geom_errorbar(aes(ymin = ood_score_mean-ood_score_se, ymax = ood_score_mean+ood_score_se), width=0.25, size=0.35) +
#   # geom_line(size=0.35, alpha=0.5)+
#   # coord_cartesian(ylim=c(0.238, 0.34))+
#   # labs(y='MCS fraction\nto train set', x='Uncertainty bin') +
#   scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
#   scale_color_manual(values = c('#97a4ab','#efc57b')) + 
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))



# comparison_variables = c('')
# target_names


# # CHEMBL1871_Ki has the strong positive correlation between OOD score an uncertainty (0.6226, highest)
# ggplot(subset(jvae_reconstruction, dataset == 'CHEMBL1871_Ki'), aes(y=ood_score, x=y_unc, color = split)) +
#   geom_point(alpha=0.5) +
#   geom_smooth(method = lm, se = FALSE)+
#   labs(y='scoreOOD', x='Uncertainty') +
#   scale_color_manual(values = c('#97a4ab','#efc57b')) +
#   default_theme + theme(legend.position = 'none', plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
# 
#   # CHEMBL2047_EC50 has a strong negative correlation between OOD score and uncertainty (-0.565)
# ggplot(subset(jvae_reconstruction, dataset == 'CHEMBL2047_EC50'), aes(y=ood_score, x=y_unc, color = split)) +
#   geom_point(alpha=0.5) +
#   geom_smooth(method = lm, se = FALSE) +
#   labs(y='scoreOOD', x='Uncertainty') +
#   scale_color_manual(values = c('#97a4ab','#efc57b')) +
#   default_theme + theme(legend.position = 'none', plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
# 
# ggplot(subset(jvae_reconstruction, dataset == 'CHEMBL1871_Ki'), aes(y=correct_pred, x=y_unc, color = split)) +
#   geom_point(alpha=0.5) +
#   geom_smooth(method = lm, se = FALSE)+
#   labs(y='Correctness', x='Uncertainty') +
#   scale_color_manual(values = c('#97a4ab','#efc57b')) +
#   default_theme + theme(legend.position = 'none', plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
#   
#   
#   ggplot(subset(jvae_reconstruction, dataset == 'CHEMBL2047_EC50'), aes(y=correct_pred, x=y_unc, color = split)) +
#   geom_point(alpha=0.5) +
#   geom_smooth(method = lm, se = FALSE)+
#   labs(y='Correctness', x='Uncertainty') +
#   scale_color_manual(values = c('#97a4ab','#efc57b')) +
#   default_theme + theme(legend.position = 'none', plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))



# ggplot(subset(jvae_reconstruction), aes(x=y_unc, color=correct_pred > 0.5)) +
#   geom_boxplot() + default_theme
#   
# 
# 
# 
library(plotly)

df_ = subset(jvae_reconstruction, dataset == 'CHEMBL1871_Ki')
plot_ly(df_, x=~MCSF, y=~y_unc, z=~ood_score, type="scatter3d", mode="markers", color=df_$correct_pred, size=1.5, opacity=~y_unc) %>% layout(scene = list(aspectmode='cube'))

ggplot(df_, aes(x=MCSF, y=ood_score, color=y_unc))+
  geom_point() + default_theme

# cor(subset(jvae_reconstruction, dataset == 'CHEMBL2835_Ki')$ood_score,
#     subset(jvae_reconstruction, dataset == 'CHEMBL2835_Ki')$y_unc)

# CHEMBL1871_Ki CHEMBL2835_Ki CHEMBL4616_EC50 CHEMBL2835_Ki CHEMBL2147_Ki




# 
# ggplot(subset(jvae_reconstruction), aes(y=MCSF, x=quartile_label, fill=split))+
#   geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
#                outlier.shape=NA, varwidth = FALSE, lwd=0.3, fatten=0.75) +
#   # scale_y_continuous(limit=c(0, 0.5)) +
#   coord_cartesian(ylim=c(0, 0.5))+
#   labs(y='MCS fraction\nto train set', x='OOD score quartile') +
#   scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
# 
# ggplot(jvae_reconstruction_acc, aes(y=balanced_acc, x=quartile_label, fill=split))+
#   geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
#   geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
#                outlier.shape=NA, varwidth = FALSE, lwd=0.3, fatten=0.75) +
#   coord_cartesian(ylim=c(0.2, 1))+
#   labs(y='Balanced accuracy', x='OOD score quartile') +
#   scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

#### 3 ####


# Compute the balanced accuracy per bin. Compute the mean and se over datasets
jvae_reconstruction_ood_binned = jvae_reconstruction %>% 
  group_by(dataset, quartile_ood) %>%
  summarise(
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    y_unc = mean(y_unc),
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5)),
    tpr = compute_tpr(y, 1*(y_hat>0.5)),
    precision = compute_precision(y, 1*(y_hat>0.5))
  ) %>% ungroup() %>% 
  group_by(quartile_ood) %>%
  summarise(
    ood_score_mean = mean(ood_score, na.rm = TRUE),
    ood_score_se = sd(ood_score, na.rm = TRUE),
    y_unc_mean = mean(y_unc, na.rm = TRUE),
    y_unc_se = se(y_unc, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE),
    tpr_mean = mean(tpr, na.rm = TRUE),
    tpr_se = se(tpr, na.rm = TRUE),
    precision_mean = mean(precision, na.rm = TRUE),
    precision_se = se(precision, na.rm = TRUE)
  ) %>% ungroup()


# Compute the balanced accuracy per bin. Compute the mean and se over datasets
jvae_reconstruction_mcsf_binned = jvae_reconstruction %>% 
  group_by(dataset, quartile_MCSF) %>%
  summarise(
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    y_unc = mean(y_unc),
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5)),
    tpr = compute_tpr(y, 1*(y_hat>0.5)),
    precision = compute_precision(y, 1*(y_hat>0.5))
  ) %>% ungroup() %>% 
  group_by(quartile_MCSF) %>%
  summarise(
    ood_score_mean = mean(ood_score, na.rm = TRUE),
    ood_score_se = sd(ood_score, na.rm = TRUE),
    y_unc_mean = mean(y_unc, na.rm = TRUE),
    y_unc_se = se(y_unc, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE),
    tpr_mean = mean(tpr, na.rm = TRUE),
    tpr_se = se(tpr, na.rm = TRUE),
    precision_mean = mean(precision, na.rm = TRUE),
    precision_se = se(precision, na.rm = TRUE)
  ) %>% ungroup()


# Compute the balanced accuracy per bin. Compute the mean and se over datasets
jvae_reconstruction_unc_binned = jvae_reconstruction %>% 
  group_by(dataset, quartile_unc) %>%
  summarise(
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    y_unc = mean(y_unc),
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5)),
    tpr = compute_tpr(y, 1*(y_hat>0.5)),
    precision = compute_precision(y, 1*(y_hat>0.5))
  ) %>% ungroup() %>% 
  group_by(quartile_unc) %>%
  summarise(
    ood_score_mean = mean(ood_score, na.rm = TRUE),
    ood_score_se = sd(ood_score, na.rm = TRUE),
    y_unc_mean = mean(y_unc, na.rm = TRUE),
    y_unc_se = se(y_unc, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE),
    tpr_mean = mean(tpr, na.rm = TRUE),
    tpr_se = se(tpr, na.rm = TRUE),
    precision_mean = mean(precision, na.rm = TRUE),
    precision_se = se(precision, na.rm = TRUE)
  ) %>% ungroup()


# Compute the balanced accuracy per bin. Compute the mean and se over datasets
jvae_reconstruction_z_binned = jvae_reconstruction %>% 
  group_by(dataset, quartileZ) %>%
  summarise(
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    y_unc = mean(y_unc),
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5)),
    tpr = compute_tpr(y, 1*(y_hat>0.5)),
    precision = compute_precision(y, 1*(y_hat>0.5))
  ) %>% ungroup() %>% 
  group_by(quartileZ) %>%
  summarise(
    ood_score_mean = mean(ood_score, na.rm = TRUE),
    ood_score_se = sd(ood_score, na.rm = TRUE),
    y_unc_mean = mean(y_unc, na.rm = TRUE),
    y_unc_se = se(y_unc, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE),
    tpr_mean = mean(tpr, na.rm = TRUE),
    tpr_se = se(tpr, na.rm = TRUE),
    precision_mean = mean(precision, na.rm = TRUE),
    precision_se = se(precision, na.rm = TRUE)
  ) %>% ungroup()



# Compute the balanced accuracy per bin. Compute the mean and se over datasets
jvae_reconstruction_E_binned = jvae_reconstruction %>% 
  group_by(dataset, quartileE) %>%
  summarise(
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    y_unc = mean(y_unc),
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5)),
    tpr = compute_tpr(y, 1*(y_hat>0.5)),
    precision = compute_precision(y, 1*(y_hat>0.5))
  ) %>% ungroup()

jvae_reconstruction_E_binned$tpr[is.na(jvae_reconstruction_E_binned$tpr)] = 0

jvae_reconstruction_E_binned = jvae_reconstruction_E_binned %>% 
  group_by(quartileE) %>%
  summarise(
    ood_score_mean = mean(ood_score, na.rm = TRUE),
    ood_score_se = sd(ood_score, na.rm = TRUE),
    y_unc_mean = mean(y_unc, na.rm = TRUE),
    y_unc_se = se(y_unc, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE),
    tpr_mean = mean(tpr, na.rm = TRUE),
    tpr_se = se(tpr, na.rm = TRUE),
    precision_mean = mean(precision, na.rm = TRUE),
    precision_se = se(precision, na.rm = TRUE)
  ) %>% ungroup()

ggplot(jvae_reconstruction_E_binned, aes(y=tpr_mean, x=quartileE))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = tpr_mean-tpr_se, ymax = tpr_mean+tpr_se), width=0.25, size=0.35, color='#577788') +
  # coord_cartesian(ylim=c(0.0, 1))+
  labs(y='Hit rate', x='E') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

ggplot(jvae_reconstruction_z_binned, aes(y=tpr_mean, x=quartileZ))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = tpr_mean-tpr_se, ymax = tpr_mean+tpr_se), width=0.25, size=0.35, color='#577788') +
  # coord_cartesian(ylim=c(0.0, 1))+
  labs(y='Hit rate', x='Z') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

ggplot(jvae_reconstruction_sdc_binned, aes(y=tpr_mean, x=quartileSDC))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = tpr_mean-tpr_se, ymax = tpr_mean+tpr_se), width=0.25, size=0.35, color='#577788') +
  # coord_cartesian(ylim=c(0.0, 1))+
  labs(y='Hit rate', x='SDC') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))



jvae_reconstruction_mcsf_binned$quartile_MCSF = factor(jvae_reconstruction_mcsf_binned$quartile_MCSF, levels = rev(jvae_reconstruction_mcsf_binned$quartile_MCSF))
fig3a = ggplot(jvae_reconstruction_mcsf_binned, aes(y=tpr_mean, x=quartile_MCSF))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = tpr_mean-tpr_se, ymax = tpr_mean+tpr_se), width=0.25, size=0.35, color='#577788') +
  scale_x_discrete(labels = as.character(1:8)) +
  coord_cartesian(ylim=c(0.50, 0.95))+
  labs(y='Hit rate', x='Novelty (binned)') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig3b = ggplot(jvae_reconstruction_ood_binned, aes(y=tpr_mean, x=quartile_ood))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = tpr_mean-tpr_se, ymax = tpr_mean+tpr_se), width=0.25, size=0.35, color='#577788') +
  coord_cartesian(ylim=c(0.50, 0.95))+
  labs(y='Hit rate', x='Unfamiliarity (binned)') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig3c = ggplot(jvae_reconstruction_unc_binned, aes(y=tpr_mean, x=quartile_unc))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = tpr_mean-tpr_se, ymax = tpr_mean+tpr_se), width=0.25, size=0.35, color='#577788') +
  coord_cartesian(ylim=c(0.50, 0.95))+
  labs(y='Hit rate', x='Uncertainty (binned)') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig3d = ggplot(jvae_reconstruction_unc_binned, aes(y=ood_score_mean, x=quartile_unc))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = ood_score_mean-ood_score_se, ymax = ood_score_mean+ood_score_se), width=0.25, size=0.35, color='#577788') +
  # coord_cartesian(ylim=c(0.50, 0.95))+
  labs(y='Unfamiliarity', x='Uncertainty (binned)') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig3abcd = plot_grid(fig3a, fig3b, fig3c, fig3d, ncol=4, labels = c('a', 'b', 'c', 'd'), label_size = 10)

####



jvae_reconstruction



jvae_reconstruction_mcsf_binned$quartile_MCSF = factor(jvae_reconstruction_mcsf_binned$quartile_MCSF, levels = rev(jvae_reconstruction_mcsf_binned$quartile_MCSF))
ggplot(jvae_reconstruction_ood_binned, aes(y=balanced_acc_mean, x=quartile_ood))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35, color='#577788') +
  coord_cartesian(ylim=c(0.50, 1))+
  geom_line(data = data.frame(quartile_ood = 1:8, balanced_acc_mean = seq(1, 0.5, length.out = 8)), 
            aes(x = quartile_ood, y = balanced_acc_mean), linetype = "dashed", color = "black") +
  labs(y='Balanced accuracy', x='Unfamiliarity (binned)') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(jvae_reconstruction_unc_binned, aes(y=balanced_acc_mean, x=quartile_unc))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35, color='#577788') +
  coord_cartesian(ylim=c(0.50, 1))+
  geom_line(data = data.frame(quartile_unc = 1:8, balanced_acc_mean = seq(1, 0.5, length.out = 8)), 
            aes(x = quartile_unc, y = balanced_acc_mean), linetype = "dashed", color = "black") +
  labs(y='Balanced accuracy', x='Uncertainty (binned)') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(jvae_reconstruction_mcsf_binned, aes(y=balanced_acc_mean, x=quartile_MCSF))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35, color='#577788') +
  coord_cartesian(ylim=c(0.50, 1))+
  geom_line(data = data.frame(quartile_MCSF = 1:8, balanced_acc_mean = seq(1, 0.5, length.out = 8)), 
            aes(x = quartile_MCSF, y = balanced_acc_mean), linetype = "dashed", color = "black") +
  labs(y='Balanced accuracy', x='MCSF (binned)') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) +
  
  ggplot(jvae_reconstruction_z_binned, aes(y=balanced_acc_mean, x=quartileZ))+
  geom_point(size=0.35, color='#577788')+
  geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35, color='#577788') +
  coord_cartesian(ylim=c(0.50, 1))+
  geom_line(data = data.frame(quartileZ = 1:8, balanced_acc_mean = seq(1, 0.5, length.out = 8)), 
            aes(x = quartileZ, y = balanced_acc_mean), linetype = "dashed", color = "black") +
  labs(y='Balanced accuracy', x='z-distance (binned)') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))






jvae_reconstruction_ood_binned$name = 'unfamiliarity'
jvae_reconstruction_unc_binned$name = 'uncertainty'
jvae_reconstruction_mcsf_binned$name = 'mcsf'
jvae_reconstruction_z_binned$name = 'z'

df_binned = rbind(jvae_reconstruction_ood_binned[2:14], jvae_reconstruction_unc_binned[2:14], jvae_reconstruction_mcsf_binned[dim(jvae_reconstruction_mcsf_binned)[1]:1,][2:14], jvae_reconstruction_z_binned[2:14])
df_binned$bin = rep(1:8, 4)-1


ggplot(df_binned, aes(y=precision_mean, x=bin, color=name, fill=name))+
  geom_line(size=0.35)+
  geom_ribbon(aes(ymin = precision_mean - precision_se, ymax = precision_mean + precision_se), size=0, alpha=0.1) +
  # geom_ribbon(aes(ymin = tpr_mean - tpr_se, ymax = tpr_mean + tpr_se), size=0, alpha=0.1) +
  coord_cartesian(ylim=c(0.0, 1), xlim=c(0, 7))+
  geom_abline(slope=-1/8, intercept = 1, linetype = "dashed", color = "black") +
  labs(y='Precision', x='bins') +
  default_theme + theme(legend.position = 'right',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


ggplot(df_binned, aes(y=balanced_acc_mean, x=bin, color=name, fill=name))+
  geom_line(size=0.35)+
  geom_ribbon(aes(ymin = balanced_acc_mean - balanced_acc_se, ymax = balanced_acc_mean + balanced_acc_se), size=0, alpha=0.1) +
  # geom_ribbon(aes(ymin = tpr_mean - tpr_se, ymax = tpr_mean + tpr_se), size=0, alpha=0.1) +
  coord_cartesian(ylim=c(0.50, 1), xlim=c(0, 7))+
  geom_abline(slope=-1/8/2, intercept = 1, linetype = "dashed", color = "black") +
  labs(y='Balanced Accuracy', x='bins') +
  default_theme + theme(legend.position = 'right',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))







y_true = sample(c(1,0), 100, replace=T)
y_hat = sample(c(1,0), 100, replace=T)

compute_balanced_accuracy(y_true, y_true)


##### PCA ####
# Scaled PCA of Average MCSG of the top octile per method per dataset


data_to_bi_pca = function(df, axes = c(1, 2), loading_scaling=0.7, scale=T){
  #' Create a dataframe of eigenvectors and loadings from some data
  
  pca <- prcomp(df, scale = scale)
  
  scree = fviz_eig(pca)
  var <- facto_summarize(pca, element = "var", 
                         result = c("coord", "contrib", 'cos2'), 
                         axes = axes)
  ind <- facto_summarize(pca, element = "ind", 
                         result = c("coord", "contrib", 'cos2'), 
                         axes = axes)
  
  colnames(var)[2:3] <- c("x", "y")
  colnames(ind)[2:3] <- c("x", "y")
  
  # Scale Loadings 
  r <- min((max(ind[, "x"]) - min(ind[, "x"])/(max(var[, "x"]) - 
                                                 min(var[, "x"]))), (max(ind[, "y"]) - min(ind[, "y"])/
                                                                       (max(var[, "y"]) - min(var[, "y"]))))
  var[, c("x", "y")] <- var[, c("x", "y")] * r * loading_scaling
  
  # Merge the indivduals (eigenvectors) and variables 
  # (loading rotations, now scaled)
  var$type = rep('Loading', nrow(var))
  ind$type = rep('Score', nrow(ind))
  bi = rbind(ind, var)
  bi$cos2[bi$type == 'Score'] = 0
  bi$contrib[bi$type == 'Score'] = 0
  
  return( list('bi'=bi, 'pca'=pca, 'scree'=scree) )
}



data_to_biplot = function(pca_dat, val_var="MCSF", loading_scaling=1.6, lower_better=TRUE){
  #' convert a dataframe into pca data + best/worst scaling
  
  M_all = dcast(data = pca_dat, formula = acquisition_method~dataset, value.var = val_var)
  
  rownames(M_all) = str_to_sentence(gsub('_', ' ', unlist(M_all[1])))
  M_all = M_all[2:ncol(M_all)]
  M_all= data.frame(t(M_all))
  
  if (lower_better == TRUE){
    M_all$Best = apply(M_all, 1, FUN = min)
    M_all$Worst = apply(M_all, 1, FUN = max)
  } else {
    M_all$Best = apply(M_all, 1, FUN = max)
    M_all$Worst = apply(M_all, 1, FUN = min)
  }
  
  colnames(M_all) = gsub('9', ' - ', gsub('\\.', ' ', colnames(M_all)))
  
  
  pca_all = data_to_bi_pca(t(M_all), loading_scaling=loading_scaling, scale=F)
  scree_all = pca_all$scree
  
  pca_all$bi$name = gsub('Best', 'Best - Best', pca_all$bi$name)
  pca_all$bi$name = gsub('Worst', 'Worst - Worst', pca_all$bi$name)
  
  return(pca_all)
  
}



# Hit rate of the top octile selected of the top octile most novel molecules

# The highest predicted value
most_novel_molecules = subset(jvae_reconstruction, quartile_MCSF == '1') %>% 
  group_by(dataset) %>%
  mutate(quartile_ood = factor(ntile(ood_score, 4))
  ) %>% ungroup() %>% group_by(dataset) %>%
  mutate(quartile_unc = factor(ntile(y_unc, 4))
  ) %>% ungroup() %>% group_by(dataset) %>%
  mutate(quartileSDC = factor(ntile(sdc_ad, 4))
  ) %>% ungroup() %>% group_by(dataset) %>%
  mutate(quartileZ = factor(ntile(mean_z_dist, 4))
  ) %>% ungroup() %>% group_by(dataset) %>%
  mutate(quartileE = factor(ntile(y_E, 4))
  ) %>% ungroup()


# the best predicted (highest expected value)
best_E_octile = subset(most_novel_molecules, quartileE == '4')
best_E_octile$acquisition_method = 'highest E'

# the most familiar
best_ood_octile = subset(most_novel_molecules, quartile_ood == '1')
best_ood_octile$acquisition_method = 'least unfamiliar'

# the most certain
best_unc_octile = subset(most_novel_molecules, quartile_unc == '1')
best_unc_octile$acquisition_method = 'least uncertain'

# the highest SDC (least OOD)
best_sdc_octile = subset(most_novel_molecules, quartileSDC == '4')
best_sdc_octile$acquisition_method = 'highest SDC'

# the closest Z vectors (lowest distance to train)
best_z_octile = subset(most_novel_molecules, quartileZ == '1')
best_z_octile$acquisition_method = 'closest embedding'


pca_df = rbind(best_E_octile, best_ood_octile, best_unc_octile, best_sdc_octile, best_z_octile)
pca_df = subset(pca_df) %>% group_by(dataset, acquisition_method) %>%
  summarise(
    MCSF = mean(MCSF),
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5)),
    tpr = compute_tpr(y, 1*(y_hat>0.5))
  ) %>% ungroup() 

# 
# ggplot(pca_df, aes(y = tpr, x = acquisition_method))+
#   # labs(y='Scaffold similarity\nto train set', x='Dataset split') +
#   geom_point(size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
#   geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
#                varwidth = FALSE, lwd=0.3, fatten=0.75) +
#   # scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 

pca_df$tpr[is.na(pca_df$tpr)] = 0
pca_df$balanced_acc[is.na(pca_df$balanced_acc)] = 0

pca_df = subset(pca_df, dataset %in% subset(data.frame(table(pca_df$dataset)), Freq == 5)$Var1)


pca_all = data_to_biplot(pca_df, val_var="tpr", lower_better = F)
bi_all = pca_all$bi

# Get xy the coordinates for the best and worst points
best = unlist(subset(bi_all, name == 'Best - Best')[c(2,3)])
worst = unlist(subset(bi_all, name == 'Worst - Worst')[c(2,3)])

x_axis_label = paste0('PC1 (',round(pca_all$scree$data$eig[1],1),'%)')
y_axis_label = paste0('PC2 (',round(pca_all$scree$data$eig[2],1),'%)')

# Make the actual plot
fig3f = ggplot(bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y ), shape = 21,  size = 1, alpha = ifelse(bi_all$type == 'Score', 0.8, 0), color = "black", stroke=0.1, fill = '#577788') +
  geom_segment(aes(x = worst[1], y = worst[2], xend = best[1], yend = best[2]),
               linetype='solid',  alpha = 0.005, colour='#577788', size=0.75) +
  geom_text_repel(aes(label = name), alpha = ifelse(bi_all$type == 'Score', 1, 0), 
                  size = 2, segment.size = 0.25, force = 30, max.iter = 1505, 
                  max.overlaps = 30, show.legend = FALSE) +
  labs(x = x_axis_label, y = y_axis_label) +
  default_theme


fig3e = ggplot(jvae_reconstruction_ood_binned, aes(y=MCSF_mean, x=ood_score_mean))+
  geom_point() +
  labs(y='Novelty', x='Unfamiliarity') +
  coord_cartesian(xlim=c(0, 0))+
  default_theme


fig3ef = plot_grid(fig3e, fig3f, fig3e, ncol=3, labels = c('e', 'f', ''), label_size = 10, rel_widths = c(1,1.7,1.3))

fig3 = plot_grid(fig3abcd, fig3ef, ncol=1)


pdf('plots/figpp1.pdf', width = 180/25.4, height = 90/25.4)
print(fig3)
dev.off()


# GR, Glucocorticoid receptor 
ggplot(subset(jvae_reconstruction, dataset == 'CHEMBL2034_Ki'), aes(y=MCSF, x=ood_score, color = split)) +
  geom_point(alpha=0.5) +
  # geom_smooth(method = lm, se = FALSE)+
  labs(y='MCSF', x='Unfamiliarity') +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none', plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))




# 
# # Compute the balanced accuracy per bin. Compute the mean and se over datasets
# jvae_reconstruction_ood_binned_together = jvae_reconstruction %>% 
#   group_by(dataset, split, quartile_ood) %>%
#   summarise(
#     y_unc = mean(y_unc),
#     ood_score = mean(ood_score),
#     MCSF = mean(MCSF),
#     Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train),
#     Cats_cos = mean(Cats_cos),
#     balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5))
#   ) %>% ungroup() %>% 
#   group_by(quartile_ood) %>%
#   summarise(
#     y_unc_mean = mean(y_unc, na.rm = TRUE),
#     y_unc_se = sd(y_unc, na.rm = TRUE),
#     ood_score_mean = mean(ood_score, na.rm = TRUE),
#     ood_score_se = sd(ood_score, na.rm = TRUE),
#     MCSF_mean = mean(MCSF, na.rm = TRUE),
#     MCSF_se = se(MCSF, na.rm = TRUE),
#     Tanimoto_scaffold_to_train_mean = mean(Tanimoto_scaffold_to_train, na.rm = TRUE),
#     Tanimoto_scaffold_to_train_se = se(Tanimoto_scaffold_to_train, na.rm = TRUE),
#     Cats_cos_mean = mean(Cats_cos, na.rm = TRUE),
#     Cats_cos_se = se(Cats_cos, na.rm = TRUE),
#     balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
#     balanced_acc_se = se(balanced_acc, na.rm = TRUE)
#   ) %>% ungroup()
# 
# fig3a = ggplot(jvae_reconstruction_ood_binned_together, aes(y=balanced_acc_mean, x=quartile_ood))+
#   geom_point(size=0.35, color='#577788')+
#   geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35, color='#577788') +
#   geom_line(size=0.35, alpha=0.5)+
#   coord_cartesian(ylim=c(0.5, 0.9))+
#   labs(y='Balanced accuracy', x='OOD score bin') +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# jvae_reconstruction_unc_binned_together = jvae_reconstruction %>% 
#   group_by(dataset, split, quartile_unc) %>%
#   summarise(
#     y_unc = mean(y_unc),
#     ood_score = mean(ood_score),
#     MCSF = mean(MCSF),
#     Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train),
#     Cats_cos = mean(Cats_cos),
#     balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5))
#   ) %>% ungroup() %>% 
#   group_by(quartile_unc) %>%
#   summarise(
#     y_unc_mean = mean(y_unc, na.rm = TRUE),
#     y_unc_se = sd(y_unc, na.rm = TRUE),
#     ood_score_mean = mean(ood_score, na.rm = TRUE),
#     ood_score_se = sd(ood_score, na.rm = TRUE),
#     MCSF_mean = mean(MCSF, na.rm = TRUE),
#     MCSF_se = se(MCSF, na.rm = TRUE),
#     Tanimoto_scaffold_to_train_mean = mean(Tanimoto_scaffold_to_train, na.rm = TRUE),
#     Tanimoto_scaffold_to_train_se = se(Tanimoto_scaffold_to_train, na.rm = TRUE),
#     Cats_cos_mean = mean(Cats_cos, na.rm = TRUE),
#     Cats_cos_se = se(Cats_cos, na.rm = TRUE),
#     balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
#     balanced_acc_se = se(balanced_acc, na.rm = TRUE)
#   ) %>% ungroup()
# 
# fig3b = ggplot(jvae_reconstruction_unc_binned_together, aes(y=balanced_acc_mean, x=quartile_unc))+
#   geom_point(size=0.35, color='#577788') +
#   geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35, color='#577788') +
#   geom_line(size=0.35, alpha=0.5)+
#   coord_cartesian(ylim=c(0.5, 0.9))+
#   labs(y='Balanced accuracy', x='Uncertainty bin') +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# fig3c = ggplot(jvae_reconstruction_unc_binned_together, aes(y=ood_score_mean, x=quartile_unc))+
#   geom_point(size=0.35, color='#577788') +
#   geom_errorbar(aes(ymin = ood_score_mean-ood_score_se, ymax = ood_score_mean+ood_score_se), width=0.25, size=0.35, color='#577788') +
#   geom_line(size=0.35, alpha=0.5)+
#   coord_cartesian(ylim=c(0.3, 1.1))+
#   labs(y='OOD score', x='Uncertainty bin') +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# fig3d = ggplot(jvae_reconstruction, aes(y=ood_score, x=log(y_unc))) +
#   labs(y='OOD score', x='log(uncertainty)') +
#   geom_density_2d(aes(alpha=..level..), linewidth=0.75, color='#577788') +
#   # scale_y_continuous(limit=c(0, 1.25)) +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# 
# fig3abcd = plot_grid(fig3a, fig3b, fig3c, fig3d, ncol=4, labels = c('a', 'b', 'c', 'd'), label_size = 10)
# 
# 
# 
# jvae_reconstruction_ = jvae_reconstruction %>% 
#   group_by(dataset, quartile_MCSF) %>%
#   summarise(
#     y_unc = mean(y_unc),
#     balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5))
#   ) %>% ungroup() %>% group_by(quartile_MCSF) %>%
#   summarise(
#     y_unc_mean = mean(y_unc),
#     y_unc_se = se(y_unc),
#     balanced_acc_mean = mean(balanced_acc),
#     balanced_acc_se = se(balanced_acc)
#   )
# 
# fig3e = ggplot(jvae_reconstruction_, aes(y=balanced_acc_mean, x=quartile_MCSF))+
#   geom_point(size=0.35, color='#577788')+
#   geom_errorbar(aes(ymin = balanced_acc_mean-balanced_acc_se, ymax = balanced_acc_mean+balanced_acc_se), width=0.25, size=0.35, color='#577788') +
#   coord_cartesian(ylim=c(0.5, 0.9))+
#   labs(y='Balanced accuracy', x='MCSF to train bin') +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# fig3f = ggplot(jvae_reconstruction_, aes(y=y_unc_mean, x=quartile_MCSF))+
#   geom_point(size=0.35, color='#577788')+
#   geom_errorbar(aes(ymin = y_unc_mean-y_unc_se, ymax = y_unc_mean+y_unc_se), width=0.25, size=0.35, color='#577788') +
#   # coord_cartesian(ylim=c(0.5, 0.9))+
#   labs(y='Uncertainty', x='MCSF to train bin') +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# 
# fig3efg = plot_grid(fig3e, fig3f, fig3g, ncol=4, labels = c('e', 'f', 'g'), label_size = 10)
# 
# 
# fig3 = plot_grid(fig3abcd, fig3efg, ncol=1)
# fig3
# 
# 
# 
# pdf('plots/fig3.pdf', width = 180/25.4, height = 80/25.4)
# print(fig3)
# dev.off()
# 
# 
# 
# 
# 
# figs3a = ggplot(jvae_reconstruction_unc_binned_together, aes(y=Tanimoto_scaffold_to_train_mean, x=quartile_unc))+
#   geom_point(size=0.35, color='#577788') +
#   geom_errorbar(aes(ymin = Tanimoto_scaffold_to_train_mean-Tanimoto_scaffold_to_train_se, ymax = Tanimoto_scaffold_to_train_mean+Tanimoto_scaffold_to_train_se), width=0.25, size=0.35, color='#577788') +
#   # coord_cartesian(ylim=c(0.3, 1.1))+
#   labs(y='Tani scaffold to train', x='Uncertainty bin') +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# figs3b = ggplot(jvae_reconstruction_unc_binned_together, aes(y=MCSF_mean, x=quartile_unc))+
#   geom_point(size=0.35, color='#577788') +
#   geom_errorbar(aes(ymin = MCSF_mean-MCSF_se, ymax = MCSF_mean+MCSF_se), width=0.25, size=0.35, color='#577788') +
#   # coord_cartesian(ylim=c(0.3, 1.1))+
#   labs(y='MCSF to train', x='Uncertainty bin') +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# figs3c = ggplot(jvae_reconstruction_unc_binned_together, aes(y=Cats_cos_mean, x=quartile_unc))+
#   geom_point(size=0.35, color='#577788') +
#   geom_errorbar(aes(ymin = Cats_cos_mean-Cats_cos_se, ymax = Cats_cos_mean+Cats_cos_se), width=0.25, size=0.35, color='#577788') +
#   # coord_cartesian(ylim=c(0.3, 1.1))+
#   labs(y='MCSF to train', x='Uncertainty bin') +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))
# 
# plot_grid(figs3a, figs3b, figs3c, ncol=4, labels = c('a', 'b', 'c'), label_size = 10)







