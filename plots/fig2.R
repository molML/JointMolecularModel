
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
  # panel.border = element_rect(colour = "#101e25", size = 0.75, fill = NA),
  panel.border = element_blank(),
  # axis.line.x.bottom=element_blank(),
  # axis.line.y.left=element_blank(),
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


cols = c('#577788','#97a4ab','#ef9d43','#efc57b', '#578d88', '#99beae')

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


df_datasets = df %>% group_by(split, dataset) %>%
  summarize(Tanimoto_to_train=mean(Tanimoto_to_train),
            Tanimoto_scaffold_to_train=mean(Tanimoto_scaffold_to_train),
            Cats_cos=mean(Cats_cos),
            MCSF=mean(MCSF))


fig2a = ggplot(df_datasets, aes(y = Tanimoto_scaffold_to_train, x = split, fill=split))+
  labs(y='Scaffold similarity\nto train set', x='Dataset split') +
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

fig2b = ggplot(df_datasets, aes(y = MCSF, x = split, fill=split))+
  labs(y='MCS fraction\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig2c = ggplot(df_datasets, aes(y = Cats_cos, x = split, fill=split))+
  labs(y='CATS similarity\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))



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

fig2d = ggplot(subset(df_predictions, split != 'Train'), aes(x = method, y = balanced_accuracy, fill = split)) +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_x_discrete(labels = c("CATS\nRF", "ECFP\nRF", "ECFP\nMLP", "SMILES\nMLP", "JMM")) +
  labs(x = '', y = 'Balanced Accuracy') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(plot.margin = unit(c(0.2, 0.2, -0.075, 0.2), "cm"), legend.position = 'none')



jvae_reconstruction = subset(df, method == "SMILES_JMM" & split != 'Train') %>%  #  | method == "SMILES_JMM"
  group_by(split, dataset, method, smiles) %>%
  summarize(
    ood_score = mean(ood_score),
    MCSF = mean(MCSF),
    smiles_entropy = mean(smiles_entropy))

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

fig2f = ggplot(jvae_reconstruction, aes(y=ood_score, x=MCSF)) + 
  labs(y='OOD score', x='MCS fraction to train set') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_y_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig2g = ggplot(jvae_reconstruction, aes(y=ood_score, x=smiles_entropy)) + 
  labs(y='OOD score', x='SMILES complexity') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_y_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


target_names = data.frame(id = c("PPARG", "Ames_mutagenicity", "ESR1_ant", "TP53", "CHEMBL1871_Ki","CHEMBL218_EC50","CHEMBL244_Ki","CHEMBL236_Ki","CHEMBL234_Ki","CHEMBL219_Ki","CHEMBL238_Ki","CHEMBL4203_Ki","CHEMBL2047_EC50","CHEMBL4616_EC50","CHEMBL2034_Ki","CHEMBL262_Ki","CHEMBL231_Ki","CHEMBL264_Ki","CHEMBL2835_Ki","CHEMBL2971_Ki","CHEMBL237_EC50","CHEMBL237_Ki","CHEMBL233_Ki","CHEMBL4792_Ki","CHEMBL239_EC50","CHEMBL3979_EC50","CHEMBL235_EC50","CHEMBL4005_Ki","CHEMBL2147_Ki","CHEMBL214_Ki","CHEMBL228_Ki","CHEMBL287_Ki","CHEMBL204_Ki","CHEMBL1862_Ki"),
                          name = c("PPARyl", "Ames", "ESR1", "TP53", "AR","CB1","FX","DOR","D3R","D4R","DAT","CLK4","FXR","GHSR","GR","GSK3","HRH1","HRH3","JAK1","JAK2","KOR (a)","KOR (i)","MOR","OX2R","PPARa","PPARym","PPARd","PIK3CA","PIM1","5-HT1A","SERT","SOR","Thrombin","ABL1"))
# PPARG - PPARy (LitPCBA) agonist (according to the paper, the website says inhibitors (antagonists))
# CHEMBL3979_EC50 - PPARy (MoleculeACE)

jvae_reconstruction$dataset_name = target_names$name[match(jvae_reconstruction$dataset, target_names$id)]
ood_score_order = (subset(jvae_reconstruction, split == 'OOD') %>% group_by(dataset_name) %>% summarize(ood_score = mean(ood_score)) %>% arrange(-ood_score) %>% distinct(dataset_name))$dataset_name
jvae_reconstruction$dataset_name = factor(jvae_reconstruction$dataset_name, levels=ood_score_order)

fig2h = ggplot(jvae_reconstruction) +
  geom_density_ridges(aes(x = ood_score, y = dataset_name, fill = split), alpha = 0.75, linewidth=0.35) +
  labs(x="OOD score", y='') +
  scale_fill_manual(values = c('#577788','#efc57b')) +
  scale_x_continuous(limit=c(-1.8, 2.2)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

fig2abc_ = plot_grid(fi
                     g2a, fig2b, fig2c, ncol=3, labels = c('a', 'b', 'c'), label_size = 10)
fig2de_ = plot_grid(fig2d, fig2e, ncol=2, rel_widths = c(2.25,1), labels = c('d', 'e'), label_size = 10)
fig2fg_ = plot_grid(fig2f, fig2g, ncol=2, labels = c('f', 'g'), label_size = 10)

fig2abcdefg_ = plot_grid(fig2abc_, fig2de_, fig2fg_, ncol=1)

fig2 = plot_grid(fig2abcdefg_, fig2h, label_size = 10, ncol=2, rel_widths = c(1, 0.8), labels = c('', 'h'))
fig2

pdf('plots/fig2__.pdf', width = 180/25.4, height = 130/25.4)
print(fig2)
dev.off()




# # Find the size of the train set for every dataset
# dataset_train_size = subset(df_summary, split == 'Train') %>% distinct(smiles_id_mean, dataset, .keep_all = TRUE) %>% group_by(split, dataset) %>% summarize(set_size = length(split))
# jvae_dataset_size <- jvae_reconstruction %>%
#   group_by(split, dataset) %>%
#   summarize(
#     set_size = length(split),
#     ood_score = mean(ood_score_mean),
#     edit_distance = mean(edit_distance),
#   )
# jvae_dataset_size$train_set_size = dataset_train_size$set_size[match(jvae_dataset_size$dataset, dataset_train_size$dataset)]
# 
# fig2h = ggplot(jvae_dataset_size, aes(y = ood_score, x = train_set_size, color=split, fill=split))+
#   geom_point(size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
#   labs(y="", x='Train set size') +
#   geom_smooth(method='lm', se = T, fullrange=TRUE, alpha=0.1, linewidth=0.5) +
#   scale_y_continuous(limit=c(-0.5, 1.5)) +
#   scale_color_manual(values = c('#577788','#efc57b')) +
#   scale_fill_manual(values = c('#577788','#efc57b')) +
#   default_theme + theme(legend.position = 'none',
#                         plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))



