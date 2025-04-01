# This file plots the supplementary figure sFig6 of the paper
#
# Derek van Tilborg
# Eindhoven University of Technology
# March 2025

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
library(reshape2)
library(stringr)
library(ggpubr)
library(scatterplot3d)

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

descr_cols = list(cols = c('#efc57b','#ef9d43','#b75a33',
                           '#97a4ab', '#577788',
                           '#99beae','#578d88', 
                           '#ffffff', '#101e25', '#101e25'),
                  descr =  c("Least uncertain", "Least unfamiliar", "Certain unfamiliar", 
                             "Least uncertain novel cores", "Least unfamiliar novel cores",
                             "Least uncertain novel cats", "Least unfamiliar novel cats", 
                             "", 'Best', 'Worst'))



se <- function(x, na.rm = FALSE) {sd(x, na.rm=na.rm) / sqrt(sum(1*(!is.na(x))))}


# Load the data and change some names/factors
# setwd("~/Dropbox/PycharmProjects/JointMolecularModel")


df_3_efg <- read_csv('results/screening_mols_properties_top50.csv')
df_3_abc <- read_csv('plots/data/df_3.csv')

# Only keep the relevant ranking methods
df_3_efg = subset(df_3_efg, ranking_method %in% c('utopia_dist_E_min_unc',
                                                  'utopia_dist_E_min_ood',
                                                  'utopia_dist_E_min_unc_min_ood'
))

df_3_efg$ranking_method = gsub('utopia_dist_E_min_unc_min_ood', 'Most reliable', df_3_efg$ranking_method)
df_3_efg$ranking_method = gsub('utopia_dist_E_min_unc', 'Least uncertain', df_3_efg$ranking_method)
df_3_efg$ranking_method = gsub('utopia_dist_E_min_ood', 'Least unfamiliar', df_3_efg$ranking_method)

# Order factors
df_3_efg$ranking_method = factor(df_3_efg$ranking_method, levels = c(
  'Least uncertain',
  'Least unfamiliar',
  'Most reliable'
))



sfig6a = ggplot(df_3_efg, aes(y = `Hit rate`, x = ranking_method, fill=ranking_method))+
  labs(y='Hit rate', x='', title='') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#efc57b', '#ef9d43', '#b75a33')) +
  scale_color_manual(values = c('#efc57b', '#ef9d43', '#b75a33')) +
  scale_y_continuous(limit=c(0.25, 1)) +
  # coord_flip() +
  default_theme + theme(legend.position = 'none',
                        # axis.text.y=element_blank(),
                        axis.text.x=element_text(angle=45, hjust=1),
                        plot.margin = unit(c(0.2, 0.2, -0.6, 0), "cm"))


wilcox.test(subset(df_3_efg, ranking_method == 'Least uncertain')$`Hit rate`,
            subset(df_3_efg, ranking_method == 'Least unfamiliar')$`Hit rate`, paired=TRUE, alternative = 'two.sided')

wilcox.test(subset(df_3_efg, ranking_method == 'Least uncertain')$`Hit rate`,
            subset(df_3_efg, ranking_method == 'Most reliable')$`Hit rate`, paired=TRUE, alternative = 'two.sided')

wilcox.test(subset(df_3_efg, ranking_method == 'Most reliable')$`Hit rate`,
            subset(df_3_efg, ranking_method == 'Least unfamiliar')$`Hit rate`, paired=TRUE, alternative = 'two.sided')


sfig6b = ggplot(df_3_efg, aes(y = Enrichment, x = ranking_method, fill=ranking_method))+
  labs(y='Enrichment', x='', title='') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#efc57b', '#ef9d43', '#b75a33')) +
  scale_color_manual(values = c('#efc57b', '#ef9d43', '#b75a33')) +
  # scale_y_continuous(limit=c(0.125, 0.5)) +
  # coord_flip() +
  default_theme + theme(legend.position = 'none',
                        # axis.text.y=element_blank(),
                        axis.text.x=element_text(angle=45, hjust=1),
                        plot.margin = unit(c(0.2, 0.2, -0.6, 0), "cm"))

# 
wilcox.test(subset(df_3_efg, ranking_method == 'Least uncertain')$Enrichment,
            subset(df_3_efg, ranking_method == 'Least unfamiliar')$Enrichment, paired=TRUE, alternative = 'two.sided')

wilcox.test(subset(df_3_efg, ranking_method == 'Least uncertain')$Enrichment,
            subset(df_3_efg, ranking_method == 'Most reliable')$Enrichment, paired=TRUE, alternative = 'two.sided')

wilcox.test(subset(df_3_efg, ranking_method == 'Most reliable')$Enrichment,
            subset(df_3_efg, ranking_method == 'Least unfamiliar')$Enrichment, paired=TRUE, alternative = 'two.sided')


sfig6 = plot_grid(sfig6a, sfig6b, ncol=2, labels = c('a', 'b'), label_size = 10)

# # save to pdf
pdf('plots/figures/sfig6.pdf', width = 45/25.4, height = 45/25.4)
print(sfig6)
dev.off()




