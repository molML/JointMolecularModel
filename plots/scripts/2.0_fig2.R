# This file plots the main results (Fig2) of the paper and performs statistical tests
#
# Derek van Tilborg
# Eindhoven University of Technology
# January 2024

# loading some libraries
library(readr)
library(ggplot2)
library(dplyr)
library(cowplot)
library(ggridges)
library(viridis)
library(hrbrthemes)
library(patchwork)
library(ggrepel)
library(factoextra)


#### Theme ####

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


#### Load data ####

# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointMolecularModel")


df_2abc <- read_csv('plots/data/df_2abc.csv')
df_2d <- read_csv('plots/data/df_2d.csv')
df_2efg <- read_csv('plots/data/df_2efg.csv')
df_2h <- read_csv('plots/data/df_2h.csv')


# order factors in datasets
df_2abc$split = factor(df_2abc$split, levels = c('Train', 'Test', 'OOD'))

df_2d$split = factor(df_2d$split, levels = c('Train', 'Test', 'OOD'))
df_2d$method = factor(df_2d$method, levels = c("CATS_RF", "ECFP_RF", "ECFP_MLP", "SMILES_MLP", "SMILES_JMM", 'SMILES_AE'))

df_2h$split = factor(df_2h$split, levels = c('Train', 'Test', 'OOD'))
df_2h$quartile_ood = factor(df_2h$quartile_ood)

df_2efg$split = factor(df_2efg$split, levels = c('Train', 'Test', 'OOD'))

# Order data based on OOD score. This is not to imply any real order (datasets are independent), but just to make it less chaotic visually
ood_score_order = (subset(df_2efg, split == 'OOD') %>% group_by(dataset_name) %>% summarize(ood_score = mean(ood_score)) %>% arrange(-ood_score) %>% distinct(dataset_name))$dataset_name
df_2efg$dataset_name = factor(df_2efg$dataset_name, levels=ood_score_order)


#### 2abc ####
# Here we describe the relative molecular composition of the data splits per dataset

# boxplot of scaffold similarity
fig2a = ggplot(df_2abc, aes(y = Tanimoto_scaffold_to_train, x = split, fill=split))+
  labs(y='Scaffold similarity\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# Wilcoxon signed-rank test
wx = wilcox.test(subset(df_2abc, split == 'OOD')$Tanimoto_scaffold_to_train,
                 subset(df_2abc, split == 'Test')$Tanimoto_scaffold_to_train,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2a, Test - OOD: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_2abc, split == 'Train')$Tanimoto_scaffold_to_train,
                 subset(df_2abc, split == 'Test')$Tanimoto_scaffold_to_train,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2a, Train - Test: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))


# boxplot of MCS similarity
fig2b = ggplot(df_2abc, aes(y = MCSF, x = split, fill=split))+
  labs(y='MCS fraction\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# Wilcoxon signed-rank test
wx = wilcox.test(subset(df_2abc, split == 'OOD')$MCSF,
                 subset(df_2abc, split == 'Test')$MCSF,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2b, Test - OOD: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_2abc, split == 'Train')$MCSF,
                 subset(df_2abc, split == 'Test')$MCSF,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2b, Train - Test: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))


# boxplot of pharmacophore similarity
fig2c = ggplot(df_2abc, aes(y = Cats_cos, x = split, fill=split))+
  labs(y='CATS similarity\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# Wilcoxon signed-rank test
wx = wilcox.test(subset(df_2abc, split == 'OOD')$Cats_cos,
            subset(df_2abc, split == 'Test')$Cats_cos,
            paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2c, Test - OOD: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_2abc, split == 'Train')$Cats_cos,
                 subset(df_2abc, split == 'Test')$Cats_cos,
                 paired=TRUE, alternative = 'two.sided')
print(paste0('Fig2c, Train - Test: ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))


#### 2d ####
# Here we describe model predictive performance

# plot a boxplot of predictive performance per method per data split
fig2d = ggplot(df_2d, aes(x = method, y = balanced_accuracy, fill = split)) +
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
for (method_ in unique(df_2d$method)){
  
  wx = wilcox.test(subset(df_2d, method == method_ & split == 'OOD')$balanced_accuracy,
                   subset(df_2d, method == method_ & split == 'Test')$balanced_accuracy,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0(method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}


wx = wilcox.test(subset(df_2d, method == 'SMILES_MLP' & split == 'OOD')$balanced_accuracy,
            subset(df_2d, method == 'SMILES_JMM' & split == 'OOD')$balanced_accuracy,
            paired=TRUE, alternative = 'two.sided')
print(paste0('yes/no JMM decoder', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))


mean(subset(df_2d, method == 'SMILES_JMM' & split == 'Test')$balanced_accuracy)
se(subset(df_2d, method == 'SMILES_JMM' & split == 'Test')$balanced_accuracy)


mean(subset(df_2d, method == 'ECFP_MLP' & split == 'Test')$balanced_accuracy)
se(subset(df_2d, method == 'ECFP_MLP' & split == 'Test')$balanced_accuracy)


#### 2e ####
# Here we describe the OOD score globally

# violin + box plot of OOD score distribution in the ID and OOD molecules
fig2e = ggplot(df_2efg, aes(y = ood_score, x = split, fill=split))+
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
ks = ks.test(subset(df_2efg, split == 'OOD')$ood_score,
             subset(df_2efg, split == 'Test')$ood_score,
             alternative="two.sided")
print(paste0('fig2e KS test: ', ifelse(ks$p.value < 0.05, '*', 'n.s.'),' - ',  ks$p.value))


#### 2f ####
# Here we describe the OOD score in a dataset specific manner

# ridge plot showing the distribution of the OOD score on OOD and ID molecules
fig2f = ggplot(df_2efg) +
  geom_density_ridges(aes(x = ood_score, y = dataset_name, fill = split), alpha = 0.75, linewidth=0.35) +
  labs(x="OOD score", y='') +
  scale_fill_manual(values = c('#577788','#efc57b')) +
  scale_x_continuous(limit=c(-1.8, 2.2)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

# perform Kolmogorov-Smirnov Tests on every dataset to test if they the distributions are statistically different
print('fig2f KS tests:')
for (dataset_ in unique(df_2efg$dataset_name)){
  ks = ks.test(subset(df_2efg, dataset_name == dataset_ & split == 'OOD')$ood_score,
               subset(df_2efg, dataset_name == dataset_ & split == 'Test')$ood_score,
               alternative="two.sided")
  print(paste0(dataset_, ': ', ifelse(ks$p.value < 0.05, '*', 'n.s.'),' - ',  ks$p.value))
}


#### 2gh ####
# Here we describe the relationship between OOD score and distance to the train data

# 2D distribution of molecular similarity to the train set vs OOD score
fig2g = ggplot(df_2efg, aes(x=ood_score, y=MCSF)) + 
  labs(x='OOD score', y='MCS fraction\nto train set') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

# Plot showing the relationship between data distance and binned OOD scores
fig2h = ggplot(df_2h, aes(y=MCSF_mean, x=quartile_ood, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = MCSF_mean-MCSF_se, ymax = MCSF_mean+MCSF_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  coord_cartesian(ylim=c(0.238, 0.34))+
  labs(y='MCS fraction\nto train set', x='OOD score bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


#### Fig 2 ####
# combining all plots into subplots
fig2abc_ = plot_grid(fig2a, fig2b, fig2c, ncol=3, labels = c('a', 'b', 'c'), label_size = 10)
fig2de_ = plot_grid(fig2d, fig2e, ncol=2, rel_widths = c(2.25,1), labels = c('d', 'e'), label_size = 10)
fig2gh_ = plot_grid(fig2g, fig2h, ncol=2, labels = c('g', 'h'), label_size = 10)
fig2abcdegh_ = plot_grid(fig2abc_, fig2de_, fig2gh_, ncol=1)
fig2 = plot_grid(fig2abcdegh_, fig2f, label_size = 10, ncol=2, rel_widths = c(1, 0.8), labels = c('', 'f'))
fig2

# save to pdf
pdf('plots/figures/fig2.pdf', width = 180/25.4, height = 130/25.4)
print(fig2)
dev.off()

