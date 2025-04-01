# This file plots the supplementary figure sFig3 of the paper
#
# Derek van Tilborg
# Eindhoven University of Technology
# January 2025

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

df_2efg <- read_csv('plots/data/df_2efg.csv')
df_2h <- read_csv('plots/data/df_2h.csv')

# order factors in datasets
df_2efg$split = factor(df_2efg$split, levels = c('Train', 'Test', 'OOD'))

df_2h$split = factor(df_2h$split, levels = c('Train', 'Test', 'OOD'))
df_2h$quartile_ood = factor(df_2h$quartile_ood)


sfig3a = ggplot(df_2efg, aes(x=ood_score, y=Cats_cos)) + 
  labs(x='U(x)', y='Pharmacophore similarity\nto train set') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_y_continuous(limit=c(0.05, 0.5)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig3b = ggplot(df_2h, aes(y=Cats_cos_mean, x=quartile_ood, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = Cats_cos_mean-Cats_cos_se, ymax = Cats_cos_mean+Cats_cos_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  coord_cartesian(ylim=c(0.25, 0.42))+
  labs(y='Pharmacophore similarity\nto train set', x='U(x) binned') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


sfig3c = ggplot(df_2efg, aes(x=ood_score, y=Tanimoto_scaffold_to_train)) + 
  labs(x='U(x)', y='Scaffold similarity\nto train set') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_y_continuous(limit=c(0.05, 0.7)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig3d = ggplot(df_2h, aes(y=Tanimoto_scaffold_to_train_mean, x=quartile_ood, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = Tanimoto_scaffold_to_train_mean-Tanimoto_scaffold_to_train_se, ymax = Tanimoto_scaffold_to_train_mean+Tanimoto_scaffold_to_train_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  coord_cartesian(ylim=c(0.3, 0.55))+
  labs(y='Scaffold similarity\nto train set', x='U(x) binned') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


#### Sup Fig S3  ####
# combining all plots into subplots
sfig3abcd = plot_grid(sfig3a, sfig3b, sfig3c, sfig3d, ncol=4, labels = c('a', 'b', 'c', 'd'), label_size = 10)
sfig3abcd

# save to pdf
pdf('plots/figures/sfig3.pdf', width = 180/25.4, height = 40/25.4)
print(sfig3abcd)
dev.off()
