# This file plots the main results (fig3) of the paper
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
setwd("~/Dropbox/PycharmProjects/JointMolecularModel")

df_3_abc <- read_csv('plots/data/df_3_abc.csv')
df_3_abc$split = factor(df_3_abc$split, levels = c('Test', 'OOD', 'Library'))

df_3_d <- read_csv('plots/data/df_3_d.csv')
df_3_e <- read_csv('plots/data/df_3_e.csv')


#### Distributions ####

# Ridge plot similarity
fig3a = ggplot(df_3_abc) +
  geom_density_ridges(aes(x = Tanimoto_to_train, y=split, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="Similarity to train", y='') +
  scale_fill_manual(values = c('#577788','#efc57b', '#4E7665', '#79A188','#A7C6A5' )) +
  scale_x_continuous(limit=c(0, 0.5)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.3, 0.4, 0.2, 0), "cm"),)

ks.test(subset(df_3_abc, split == 'Library')$Tanimoto_to_train,
        subset(df_3_abc, split == 'Test')$Tanimoto_to_train,
        alternative="two.sided")
# D = 0.49873, p-value < 2.2e-16

ks.test(subset(df_3_abc, split == 'OOD')$Tanimoto_to_train,
        subset(df_3_abc, split == 'Test')$Tanimoto_to_train,
        alternative="two.sided")
# D = 0.2786, p-value < 2.2e-16
 

# Ridge plot uncertainty
fig3b = ggplot(df_3_abc) +
  geom_density_ridges(aes(x = y_unc_mean, y=split, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="H(x)", y='') +
  scale_fill_manual(values = c('#577788','#efc57b', '#4E7665', '#79A188','#A7C6A5' )) +
  # scale_x_continuous(limit=c(0, 0.5)) +
  default_theme + theme(legend.position = 'none',
                        axis.text.y = element_blank(),
                        plot.margin = unit(c(0.3, 0.2, 0.2, 0.2), "cm"),)

ks.test(subset(df_3_abc, split == 'Library')$y_unc_mean,
        subset(df_3_abc, split == 'Test')$y_unc_mean,
        alternative="two.sided")
# D = 0.18071, p-value < 2.2e-16

ks.test(subset(df_3_abc, split == 'OOD')$y_unc_mean,
        subset(df_3_abc, split == 'Test')$y_unc_mean,
        alternative="two.sided")
# D = 0.15546, p-value < 2.2e-16


# Ridge plot unfamiliarity
fig3c = ggplot(df_3_abc) +
  geom_density_ridges(aes(x = ood_score_mean, y=split, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="U(x)", y='') +
  scale_fill_manual(values = c('#577788','#efc57b', '#4E7665', '#79A188','#A7C6A5' )) +
  scale_x_continuous(limit=c(-2, 8)) +
  default_theme + theme(legend.position = 'none',
                        axis.text.y = element_blank(),
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

ks.test(subset(df_3_abc, split == 'Library')$ood_score_mean,
        subset(df_3_abc, split == 'Test')$ood_score_mean,
        alternative="two.sided")
# D = 0.99897, p-value < 2.2e-16

ks.test(subset(df_3_abc, split == 'OOD')$ood_score_mean,
        subset(df_3_abc, split == 'Test')$ood_score_mean,
        alternative="two.sided")
# D = 0.36773, p-value < 2.2e-16

# This maximum vertical distance between CDFs is interpreted directly as effect size.
# •	D = 0.0: identical distributions
# •	D = 0.1: small difference
# •	D = 0.3: moderate difference
# •	D = 0.5: large, clear separation
# •	D = 1.0: no overlap at all

# Tanimoto      | Test_ID to Library: 0.49873 | Test_ID to OOD: 0.27860
# Uncertainty   | Test_ID to Library: 0.18071 | Test_ID to OOD: 0.15546
# Unfamiliarity | Test_ID to Library: 0.99897 | Test_ID to OOD: 0.36773

fig3abc = plot_grid(fig3a, fig3b, fig3c, ncol=3, labels = c('a', 'b', 'c'), label_size = 10, rel_widths = c(1,0.875,0.875))


#### 2D distributions ####

fig3d = ggplot(subset(df_3_d), aes(x=ood_score, y=y_unc) ) +
  labs(x='U(x)', y='H(x)') +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.01) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=10) +
  scale_fill_gradientn(colors = rev(c('#4E7665', '#79A188','#A7C6A5'))) +
  scale_x_continuous(limit=c(1.5, 7)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.3, 0.2, 0.2, 0.2), "cm"))

fig3e = ggplot(df_3_e, aes(x=ood_score_mean, y=y_unc_mean) ) +
  labs(x='U(x)', y='H(x)') +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.01) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=10) +
  scale_fill_gradientn(colors = rev(c('#4E7665', '#79A188','#A7C6A5'))) +
  scale_x_continuous(limit=c(1.5, 7)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.3, 0.2, 0.2, 0.2), "cm"))

##### Fig 3 #####

fig3de = plot_grid(plot_spacer(), fig3d, fig3e, plot_spacer(),
                   ncol=4, labels = c('', 'd', 'e', ''), label_size = 10)

fig3 = plot_grid(fig3abc,
                 fig3de,
                 ncol=1)
fig3

# save to pdf
pdf('plots/figures/fig3.pdf', width = 180/25.4, height = 90/25.4)
print(fig3)
dev.off()


# To annotate example molecules, we calculated the utopia distance and then 
# cherry picked from the top molecules of this list for molecules that are 
# obvious enough in their structure to make the point to non-experts without 
# having to explain too much chemistry background in the paper.
# 
# To do this, you need the SMILES string of these commercial libraries, which I 
# do not provide as source data for licencing reasons. Feel free to contact me
# if you want to know more details around these example molecules and how we
# selected them.

