# This file plots the supplementary figure sFig7 of the paper
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

druglikeness <- read_csv('results/screening_libraries/druglike_descriptors.csv')
library_inference <- read_csv('results/screening_libraries/all_inference_data.csv')
library_inference = na.omit(library_inference)

library_inference <- library_inference %>%
  left_join(druglikeness, by = "smiles")


library_inference_summary = library_inference %>%
  group_by(smiles) %>%
  summarize(
    ood_score = mean(ood_score_mean),
    y_hat = mean(y_hat_mean),
    y_unc = mean(y_unc_mean),
    y_E = mean(y_E_mean),
    mean_z_dist = mean(mean_z_dist_mean),
    Tanimoto_to_train = mean(Tanimoto_to_train),
    Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train),
    Cats_cos = mean(Cats_cos),
    SA_scores = mean(SA_scores),
    MW_scores = mean(MW_scores),
    n_atoms = mean(n_atoms),
    QED_scores = mean(QED_scores)
  ) %>% ungroup()


####  sFig10 a, b - 2D distribution plots

sFig10a = ggplot(subset(library_inference_summary, split == 'Library'), aes(x=y_unc, y=Tanimoto_to_train) ) +
  labs(x='H(x)', y='Similarity to train') +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=10) +
  scale_fill_gradientn(colors = rev(c('#4E7665', '#79A188','#A7C6A5'))) +
  scale_y_continuous(limit=c(0.00, 0.175)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.3, 0.2, 0.2, 0.2), "cm"))


# 2D distribution of molecular similarity to the train set vs OOD score
sFig10b = ggplot(subset(library_inference_summary, split == 'Library'), aes(x=ood_score, y=Tanimoto_to_train) ) +
  labs(x='U(x)', y='Similarity to train') +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.01) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=10) +
  scale_fill_gradientn(colors = rev(c('#4E7665', '#79A188','#A7C6A5'))) +
  scale_y_continuous(limit=c(0.00, 0.175)) +
  scale_x_continuous(limit=c(1.5, 7)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.3, 0.2, 0.2, 0.2), "cm"))


sFig10 = plot_grid(sFig10a, sFig10b, ncol=2, labels = c('a', 'b'), label_size = 10)

# save to pdf
pdf('plots/figures/sFig7.pdf', width = 180/25.4, height = 135/25.4)
print(sFig10)
dev.off()
