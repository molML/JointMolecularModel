# This file plots the supplementary figure sFig2 of the paper
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
library(data.table)

# GGplot default theme I use
default_theme = theme(
  panel.border = element_rect(colour = "#101e25", size = 0.75, fill = NA),
  panel.background = element_blank(),
  plot.title = element_text(hjust = 0.5, face = "plain", size=8, margin = margin(b = 0)),
  axis.text.y = element_text(size=7, face="plain", colour = "#101e25"),
  axis.text.x = element_text(size=7, face="plain", colour = "#101e25"),
  axis.title.x = element_text(size=8, face="plain", colour = "#101e25"),
  axis.title.y = element_text(size=8, face="plain", colour = "#101e25"),
  axis.ticks = element_line(color="#101e25", size=0.35),
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

# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointChemicalModel")

df_2efg <- read_csv('plots/data/df_2efg.csv')


sfig2a = ggplot(df_2efg, aes(x=ood_score, y=smiles_entropy)) + 
  labs(x='OOD score', y='SMILES complexity') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig2b = ggplot(df_2efg, aes(x=ood_score, y=molecule_entropy)) + 
  labs(x='OOD score', y='Graph complexity') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig2c = ggplot(df_2efg, aes(x=ood_score, y=bottcher)) + 
  labs(x='OOD score', y='Bottcher complexity') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig2d = ggplot(df_2efg, aes(x=ood_score, y=bertz)) + 
  labs(x='OOD score', y='Bertz complexity') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig2e = ggplot(df_2efg, aes(x=ood_score, y=n_smiles_tokens)) + 
  labs(x='OOD score', y='n SMILES tokens') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig2f = ggplot(df_2efg, aes(x=ood_score, y=n_smiles_branches)) + 
  labs(x='OOD score', y='n SMILES branches') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig2g = ggplot(df_2efg, aes(x=ood_score, y=mol_weight)) + 
  labs(x='OOD score', y='Molecular weight (g/mol)') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig2h = ggplot(df_2efg, aes(x=ood_score, y=motifs)) + 
  labs(x='OOD score', y='n functional groups') +
  geom_density_2d(aes(color = split, alpha=..level..), linewidth=0.75) +
  scale_x_continuous(limit=c(0, 1.25)) +
  scale_color_manual(values = c('#97a4ab','#efc57b')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))

sfig2 = plot_grid(sfig2a, sfig2b, sfig2c, sfig2d, sfig2e, sfig2f, sfig2g, sfig2h, ncol=4, labels = c('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'), label_size = 10)
sfig2

# save to pdf
pdf('plots/figures/sfig2.pdf', width = 180/25.4, height = 70/25.4)
print(sfig2)
dev.off()