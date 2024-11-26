# This file plots the main results (Fig2) of the paper and performs statistical tests
#
# Derek van Tilborg
# Eindhoven University of Technology
# November 2024


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

# GGplot default theme I use
default_theme = theme(
  panel.border = element_blank(),
  panel.background = element_blank(),
  plot.title = element_text(hjust = 0.5, face = "plain", size=8, margin = margin(b = 0)),
  axis.text.y = element_text(size=7, face="plain", colour = "#101e25"),
  axis.text.x = element_text(size=7, face="plain", colour = "#101e25"),
  axis.title.x = element_text(size=8, face="plain", colour = "#101e25"),
  axis.title.y = element_text(size=8, face="plain", colour = "#101e25"),
  axis.ticks = element_line(color="#101e25", size=0.35),
  axis.line.x.bottom=element_line(color="#101e25", size=0.35),
  axis.line.y.left=element_line(color="#101e25", size=0.35),
  legend.key = element_blank(),
  legend.position = 'right',
  legend.title = element_text(size=8),
  legend.background = element_blank(),
  legend.text = element_text(size=8),
  legend.spacing.y = unit(0., 'cm'),
  legend.key.size = unit(0.25, 'cm'),
  legend.key.width = unit(0.5, 'cm'),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank())

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

# the color scheme I used throughout the paper
cols = c('#577788','#97a4ab','#ef9d43','#efc57b', '#578d88', '#99beae')


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


#### Section 1 (a, b, c) ####
# Here we describe the relative molecular composition of the data splits per dataset

# summarize the data for every dataset per split
df_datasets = df %>% group_by(split, dataset) %>%
  summarize(Tanimoto_to_train=mean(Tanimoto_to_train),
            Tanimoto_scaffold_to_train=mean(Tanimoto_scaffold_to_train),
            Cats_cos=mean(Cats_cos),
            MCSF=mean(MCSF))

# boxplot of scaffold similarity
fig2a = ggplot(df_datasets, aes(y = Tanimoto_scaffold_to_train, x = split, fill=split))+
  labs(y='Scaffold similarity\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# Wilcoxon signed-rank test
wx = wilcox.test(subset(df_datasets, split == 'OOD')$Tanimoto_scaffold_to_train,
                 subset(df_datasets, split == 'Test')$Tanimoto_scaffold_to_train,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2a, Test - OOD: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_datasets, split == 'Train')$Tanimoto_scaffold_to_train,
                 subset(df_datasets, split == 'Test')$Tanimoto_scaffold_to_train,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2a, Train - Test: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))


# boxplot of MCS similarity
fig2b = ggplot(df_datasets, aes(y = MCSF, x = split, fill=split))+
  labs(y='MCS fraction\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# Wilcoxon signed-rank test
wx = wilcox.test(subset(df_datasets, split == 'OOD')$MCSF,
                 subset(df_datasets, split == 'Test')$MCSF,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2b, Test - OOD: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_datasets, split == 'Train')$MCSF,
                 subset(df_datasets, split == 'Test')$MCSF,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2b, Train - Test: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))


# boxplot of pharmacophore similarity
fig2c = ggplot(df_datasets, aes(y = Cats_cos, x = split, fill=split))+
  labs(y='CATS similarity\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# Wilcoxon signed-rank test
wx = wilcox.test(subset(df_datasets, split == 'OOD')$Cats_cos,
            subset(df_datasets, split == 'Test')$Cats_cos,
            paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2c, Test - OOD: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_datasets, split == 'Train')$Cats_cos,
                 subset(df_datasets, split == 'Test')$Cats_cos,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2c, Train - Test: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))


#### Section 2 (d) ####
# Here we describe model predictive performance

# Summarize the data so it has metrics per dataset
df_predictions = subset(df, method != "SMILES_AE")
df_predictions <- df_predictions %>%
  group_by(split, dataset, method) %>%
  summarize(
    accuracy = mean(split_acc),
    TPR = mean(split_TPR),
    TNR = mean(split_TNR),
    balanced_accuracy = mean(split_balanced_acc),
    uncertainty = mean(y_unc)
  )

# plot a boxplot of predictive performance per method per data split
fig2d = ggplot(subset(df_predictions, split != 'Train'), aes(x = method, y = balanced_accuracy, fill = split)) +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_x_discrete(labels = c("CATS\nRF", "ECFP\nRF", "ECFP\nMLP", "SMILES\nMLP", "JMM")) +
  labs(x = '', y = 'Balanced Accuracy') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(plot.margin = unit(c(0.2, 0.2, -0.075, 0.2), "cm"), legend.position = 'none')

# Wilcoxon signed-rank test
print('fig2d Wilcoxon signed-rank tests:')
for (method_ in unique(df_predictions$method)){
  
  wx = wilcox.test(subset(df_predictions, method == method_ & split == 'OOD')$balanced_accuracy,
                   subset(df_predictions, method == method_ & split == 'Test')$balanced_accuracy,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0(method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}

# wilcox.test(subset(df_predictions, method == 'SMILES_MLP' & split == 'OOD')$balanced_accuracy,
#             subset(df_predictions, method == 'SMILES_JMM' & split == 'OOD')$balanced_accuracy,
#             paired=TRUE, alternative = 'two.sided')


#### Section 3 (e, f, g) ####
# Here we describe the OOD score globally


# summarize the data per sample (spanning all datasets) per split
jvae_reconstruction = subset(df, method == "SMILES_JMM" & split != 'Train') %>%  #  | method == "SMILES_JMM"
  group_by(split, dataset, smiles) %>%
  summarize(
    ood_score = mean(ood_score),
    correct_pred = mean(correct_pred),
    sdc_ad = mean(sdc_ad),
    y_hat = mean(as.numeric(as.character(y_hat))),
    y = mean(as.numeric(as.character(y))),
    y_unc = mean(y_unc),
    split_balanced_acc = mean(split_balanced_acc),
    Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train),
    MCSF = mean(MCSF),
    Cats_cos = mean(Cats_cos),
    smiles_entropy = mean(smiles_entropy)) %>% 
  ungroup() %>% group_by(dataset) %>%
  mutate(quartile_ood = factor(ntile(ood_score, 8)),
         quartile_unc = factor(ntile(y_hat, 8))
         ) %>% ungroup()

# violin + box plot of OOD score distribution in the ID and OOD molecules
fig2e = ggplot(subset(jvae_reconstruction), aes(y = ood_score, x = split, fill=split))+
  labs(y="OOD score", x='Dataset split') +
  # geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_violin(aes(color = split))+
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.15, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_y_continuous(limit=c(-0.5, 1.5)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# perform Kolmogorov-Smirnov Tests to test if the distributions are statistically different
ks = ks.test(subset(jvae_reconstruction, split == 'OOD')$ood_score,
             subset(jvae_reconstruction, split == 'Test')$ood_score,
             alternative="two.sided")
print(paste0('fig2e KS test: ', ifelse(ks$p.value < 0.05, '*', 'n.s.'),' - ',  ks$p.value))


#### Section 4 (f) ####
# Here we describe the OOD score in a dataset specific manner

# Change the dataset names to their target name
# LitPCBA and MoleculeACE (CHEMBL3979_EC50) both have a PPAR gamma dataset, I will label them with a reference in post processing
target_names = data.frame(id = c("PPARG", "Ames_mutagenicity", "ESR1_ant", "TP53", "CHEMBL1871_Ki","CHEMBL218_EC50","CHEMBL244_Ki","CHEMBL236_Ki","CHEMBL234_Ki","CHEMBL219_Ki","CHEMBL238_Ki","CHEMBL4203_Ki","CHEMBL2047_EC50","CHEMBL4616_EC50","CHEMBL2034_Ki","CHEMBL262_Ki","CHEMBL231_Ki","CHEMBL264_Ki","CHEMBL2835_Ki","CHEMBL2971_Ki","CHEMBL237_EC50","CHEMBL237_Ki","CHEMBL233_Ki","CHEMBL4792_Ki","CHEMBL239_EC50","CHEMBL3979_EC50","CHEMBL235_EC50","CHEMBL4005_Ki","CHEMBL2147_Ki","CHEMBL214_Ki","CHEMBL228_Ki","CHEMBL287_Ki","CHEMBL204_Ki","CHEMBL1862_Ki"),
                          name = c("PPARyl", "Ames", "ESR1", "TP53", "AR","CB1","FX","DOR","D3R","D4R","DAT","CLK4","FXR","GHSR","GR","GSK3","HRH1","HRH3","JAK1","JAK2","KOR (a)","KOR (i)","MOR","OX2R","PPARa","PPARym","PPARd","PIK3CA","PIM1","5-HT1A","SERT","SOR","Thrombin","ABL1"))
jvae_reconstruction$dataset_name = target_names$name[match(jvae_reconstruction$dataset, target_names$id)]

# Order data based on OOD score. This is not to imply any real order (datasets are independent), but just to make it less chaotic visually
ood_score_order = (subset(jvae_reconstruction, split == 'OOD') %>% group_by(dataset_name) %>% summarize(ood_score = mean(ood_score)) %>% arrange(-ood_score) %>% distinct(dataset_name))$dataset_name
jvae_reconstruction$dataset_name = factor(jvae_reconstruction$dataset_name, levels=ood_score_order)


# ridge plot showing the distribution of the OOD score on OOD and ID molecules
fig2f = ggplot(jvae_reconstruction) +
  geom_density_ridges(aes(x = ood_score, y = dataset_name, fill = split), alpha = 0.75, linewidth=0.35) +
  labs(x="OOD score", y='') +
  scale_fill_manual(values = c('#577788','#efc57b')) +
  scale_x_continuous(limit=c(-1.8, 2.2)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

# perform Kolmogorov-Smirnov Tests on every dataset to test if they the distributions are statistically different
print('fig2f KS tests:')
for (dataset_ in unique(jvae_reconstruction$dataset_name)){
  ks = ks.test(subset(jvae_reconstruction, dataset_name == dataset_ & split == 'OOD')$ood_score,
               subset(jvae_reconstruction, dataset_name == dataset_ & split == 'Test')$ood_score,
               alternative="two.sided")
  print(paste0(dataset_, ': ', ifelse(ks$p.value < 0.05, '*', 'n.s.'),' - ',  ks$p.value))
}

#### Section 5 (g, h) ####
# Here we describe the relationship between OOD score and distance to the train data

# 2D distribution of molecular similarity to the train set vs OOD score
fig2g = ggplot(jvae_reconstruction, aes(x=ood_score, y=MCSF)) + 
  labs(x='OOD score', y='MCS fraction\nto train set') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# Compute the balanced accuracy per bin. Compute the mean and se over datasets
jvae_reconstruction_ood_binned = jvae_reconstruction %>% 
  group_by(dataset, split, quartile_ood) %>%
  summarise(
    y_unc = mean(y_unc),
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
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
    Tanimoto_scaffold_to_train_mean = mean(Tanimoto_scaffold_to_train, na.rm = TRUE),
    Tanimoto_scaffold_to_train_se = se(Tanimoto_scaffold_to_train, na.rm = TRUE),
    Cats_cos_mean = mean(Cats_cos, na.rm = TRUE),
    Cats_cos_se = se(Cats_cos, na.rm = TRUE),
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE)
  ) %>% ungroup()

# Plot showing the relationship between data distance and binned OOD scores
fig2h = ggplot(jvae_reconstruction_ood_binned, aes(y=MCSF_mean, x=quartile_ood, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = MCSF_mean-MCSF_se, ymax = MCSF_mean+MCSF_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  coord_cartesian(ylim=c(0.238, 0.34))+
  labs(y='MCS fraction\nto train set', x='OOD score bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


# combining all plots into subplots
fig2abc_ = plot_grid(fig2a, fig2b, fig2c, ncol=3, labels = c('a', 'b', 'c'), label_size = 10)
fig2de_ = plot_grid(fig2d, fig2e, ncol=2, rel_widths = c(2.25,1), labels = c('d', 'e'), label_size = 10)
fig2gh_ = plot_grid(fig2g, fig2h, ncol=2, labels = c('g', 'h'), label_size = 10)
fig2abcdegh_ = plot_grid(fig2abc_, fig2de_, fig2gh_, ncol=1)
fig2 = plot_grid(fig2abcdegh_, fig2f, label_size = 10, ncol=2, rel_widths = c(1, 0.8), labels = c('', 'f'))
fig2

# save to pdf
pdf('plots/fig2.pdf', width = 180/25.4, height = 130/25.4)
print(fig2)
dev.off()

