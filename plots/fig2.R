# This file plots the main results (Fig2) of the paper
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

# boxplot of MCS similarity
fig2b = ggplot(df_datasets, aes(y = MCSF, x = split, fill=split))+
  labs(y='MCS fraction\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# boxplot of pharmacophore similarity
fig2c = ggplot(df_datasets, aes(y = Cats_cos, x = split, fill=split))+
  labs(y='CATS similarity\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# # Wilcoxon signed-rank test
# wilcox.test(subset(df_per_dataset, split == 'OOD')$Cats_cos,
#             subset(df_per_dataset, split == 'Train')$Cats_cos,
#             paired=TRUE, alternative = 'two.sided')


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


#### Section 3 (e, f, g) ####
# Here we describe the OOD score globally

# summarize the data per sample (spanning all datasets) per split
jvae_reconstruction = subset(df, method == "SMILES_JMM" & split != 'Train') %>%  #  | method == "SMILES_JMM"
  group_by(split, dataset, method, smiles) %>%
  summarize(
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    smiles_entropy = mean(smiles_entropy))

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

# 2D distribution of molecular similarity to the train set vs OOD score
fig2f = ggplot(jvae_reconstruction, aes(y=ood_score, x=MCSF)) + 
  labs(y='OOD score', x='MCS fraction to train set') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_y_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# 2D distribution of SMILES complexity vs OOD score
fig2g = ggplot(jvae_reconstruction, aes(y=ood_score, x=smiles_entropy)) + 
  labs(y='OOD score', x='SMILES complexity') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_y_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


#### Section 4 (h) ####
# Here we describe the OOD score in a dataset specific manner

# Change the dataset names to their target name and order data based on OOD score.
# This is not to imply any real order (datasets are independent), but just to make it less chaotic visually
# LitPCBA and MoleculeACE (CHEMBL3979_EC50) both have a PPAR gamma dataset, I will label them with a reference in post processing
target_names = data.frame(id = c("PPARG", "Ames_mutagenicity", "ESR1_ant", "TP53", "CHEMBL1871_Ki","CHEMBL218_EC50","CHEMBL244_Ki","CHEMBL236_Ki","CHEMBL234_Ki","CHEMBL219_Ki","CHEMBL238_Ki","CHEMBL4203_Ki","CHEMBL2047_EC50","CHEMBL4616_EC50","CHEMBL2034_Ki","CHEMBL262_Ki","CHEMBL231_Ki","CHEMBL264_Ki","CHEMBL2835_Ki","CHEMBL2971_Ki","CHEMBL237_EC50","CHEMBL237_Ki","CHEMBL233_Ki","CHEMBL4792_Ki","CHEMBL239_EC50","CHEMBL3979_EC50","CHEMBL235_EC50","CHEMBL4005_Ki","CHEMBL2147_Ki","CHEMBL214_Ki","CHEMBL228_Ki","CHEMBL287_Ki","CHEMBL204_Ki","CHEMBL1862_Ki"),
                          name = c("PPARyl", "Ames", "ESR1", "TP53", "AR","CB1","FX","DOR","D3R","D4R","DAT","CLK4","FXR","GHSR","GR","GSK3","HRH1","HRH3","JAK1","JAK2","KOR (a)","KOR (i)","MOR","OX2R","PPARa","PPARym","PPARd","PIK3CA","PIM1","5-HT1A","SERT","SOR","Thrombin","ABL1"))
jvae_reconstruction$dataset_name = target_names$name[match(jvae_reconstruction$dataset, target_names$id)]
ood_score_order = (subset(jvae_reconstruction, split == 'OOD') %>% group_by(dataset_name) %>% summarize(ood_score = mean(ood_score)) %>% arrange(-ood_score) %>% distinct(dataset_name))$dataset_name
jvae_reconstruction$dataset_name = factor(jvae_reconstruction$dataset_name, levels=ood_score_order)

# ridge plot showing the distribution of the OOD score on OOD and ID molecules
fig2h = ggplot(jvae_reconstruction) +
  geom_density_ridges(aes(x = ood_score, y = dataset_name, fill = split), alpha = 0.75, linewidth=0.35) +
  labs(x="OOD score", y='') +
  scale_fill_manual(values = c('#577788','#efc57b')) +
  scale_x_continuous(limit=c(-1.8, 2.2)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

# combining all plots into subplots
fig2abc_ = plot_grid(fig2a, fig2b, fig2c, ncol=3, labels = c('a', 'b', 'c'), label_size = 10)
fig2de_ = plot_grid(fig2d, fig2e, ncol=2, rel_widths = c(2.25,1), labels = c('d', 'e'), label_size = 10)
fig2fg_ = plot_grid(fig2f, fig2g, ncol=2, labels = c('f', 'g'), label_size = 10)
fig2abcdefg_ = plot_grid(fig2abc_, fig2de_, fig2fg_, ncol=1)
fig2 = plot_grid(fig2abcdefg_, fig2h, label_size = 10, ncol=2, rel_widths = c(1, 0.8), labels = c('', 'h'))
fig2

# save to pdf
pdf('plots/fig2.pdf', width = 180/25.4, height = 130/25.4)
print(fig2)
dev.off()
