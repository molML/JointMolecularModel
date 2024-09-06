
library(readr)
library(ggplot2)
library(dplyr)
library(cowplot)
library(ggridges)
library(viridis)
library(hrbrthemes)
library(data.table)
library(patchwork)

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
# df <- read_csv("data/datasets_with_metrics/all_datasets.csv")

df <- read_csv("results/processed/all_results_processed.csv")

df$split = gsub('train', 'Train', gsub('test', 'Test', gsub('ood', 'OOD', df$split)))
df$split = factor(df$split, levels = c('Train', 'Test', 'OOD'))
df$descriptor = toupper(df$descriptor)
df$descriptor = factor(df$descriptor, levels = c('ECFP', 'CATS', 'SMILES'))
df$model_type = toupper(df$model_type)
df$model_type = factor(df$model_type, levels = c('MLP', 'RF'))
df$method = paste0(df$descriptor, '_', df$model_type)
df$method = factor(df$method, levels = c("CATS_RF", "ECFP_RF", "ECFP_MLP", "SMILES_MLP" ))


df_per_dataset <- df %>%
  group_by(split, dataset) %>%
  summarize(across(everything(), mean, na.rm = TRUE))


fig2a = ggplot(df_per_dataset, aes(y = Tanimoto_scaffold_to_train, x = split, fill=split))+
  labs(y='Scaffold similarity\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig2b = ggplot(df_per_dataset, aes(y = MCSF, x = split, fill=split))+
  labs(y='MCS fraction\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig2c = ggplot(df_per_dataset, aes(y = Cats_cos, x = split, fill=split))+
  labs(y='CATS similarity\nto train set', x='Dataset split') +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#577788','#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


df_summary = df %>% distinct(smiles_id, descriptor, model_type, dataset, .keep_all = TRUE)
df_summary = na.omit(df_summary)
df_summary$predictions = ifelse(df_summary$y_hat_mean > 0.5, 1, 0)
df_per_dataset_per_method <- df_summary %>%
  group_by(split, dataset, method) %>%
  summarize(
    accuracy = sum(predictions == y) / n(),
    sensitivity = sum(predictions == 1 & y == 1) / sum(y == 1),
    specificity = sum(predictions == 0 & y == 0) / sum(y == 0),
    balanced_accuracy = (sensitivity + specificity) / 2,
    uncertainty = mean(y_unc_mean)
  )

df_per_dataset_per_method = subset(df_per_dataset_per_method, split != 'Train')


fig2d = ggplot(df_per_dataset_per_method, aes(x = method, y = balanced_accuracy, fill = split)) +
  geom_jitter(aes(fill=split), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.25,
               outlier.shape=NA, varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_x_discrete(labels = c("CATS\nRF", "ECFP\nRF", "ECFP\nMLP", "SMILES\nMLP")) +
  labs(x = '', y = 'Balanced Accuracy') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(plot.margin = unit(c(0.2, 0.2, -0.075, 0.2), "cm"), legend.position = 'none')


fig2abcd = plot_grid(fig2a, fig2b, fig2c, fig2d, label_size = 10, labels = c('a', 'b', 'c', 'd'), ncol=4, rel_widths = c(0.5, 0.5, 0.5, 1))


fig2e = ggplot(df_per_dataset_per_method, aes(x = accuracy, y = accuracy))+
  geom_point(alpha=0) +
  labs(x="Loss", y='Distance to pretrain set') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


df_vae <- read_csv("results/vae_pretraining/best_model/all_results.csv")
df_vae$ood = 'ID'
df_vae$ood[df_vae$dataset != 'ChEMBL'] = 'OOD'

fig2f = ggplot(df_vae, aes(y=smiles_entropy, x=log(reconstruction_loss))) + 
  labs(x="Loss", y='SMILES entropy') +
  stat_density2d(geom="density2d", aes(color = ood, alpha=..level..), size=1) +
  scale_color_manual(values = c('#577788','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


fig2g = ggplot(df_per_dataset_per_method, aes(x = accuracy, y = accuracy))+
  geom_point(alpha=0) +
  labs(x="Loss", y='Train set size') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

fig2h = ggplot(df_per_dataset_per_method, aes(x = accuracy, y = accuracy))+
  geom_point(alpha=0) +
  labs(x="Loss", y='Distance to train set') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


fig2efgh = plot_grid(fig2e, fig2f, fig2g, fig2h, label_size = 10, labels = c('e', 'f', 'g', 'h'), ncol=4, rel_widths = c(1, 1, 1, 1))



fig2i =  ggplot(df_per_dataset_per_method, aes(x = dataset, y = accuracy))+
  geom_point(alpha=0) +
  labs(x="Dataset", y='Loss') +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),
                        axis.text.x = element_text(angle = 90, hjust=1, vjust=0.5))


fig2 = plot_grid(fig2abcd, fig2efgh, fig2i, label_size = 10, labels = c('', '', 'i'), ncol=1, rel_heights = c(1, 1, 1.5))


pdf('plots/fig2.pdf', width = 180/25.4, height = 150/25.4)
print(fig2)
dev.off()






