# This file plots the supplementary figure sFig5 of the paper
#
# Derek van Tilborg
# Eindhoven University of Technology
# February 2025

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
setwd("~/Dropbox/PycharmProjects/JointChemicalModel")

df_s5_corr <- read_csv('plots/data/df_2efg.csv')
df_s5 <- read_csv('plots/data/df_2h.csv')


df_s5_corr = df_s5_corr %>%
  group_by(dataset) %>%
  summarize(
    r = cor(y_unc, ood_score, method = 'spearman')
  ) %>% ungroup()

print(paste0('r=',round(mean(df_s5_corr$r),2), '±', round(se(df_s5_corr$r), 2)))


# order factors in datasets
df_s5$split = factor(df_s5$split, levels = c('Train', 'Test', 'OOD'))
df_s5$quartile_ood = factor(df_s5$quartile_ood)

# Plot showing the relationship between data distance and binned OOD scores
sfig5 = ggplot(df_s5, aes(y=y_unc_mean, x=quartile_ood, color=split, fill=split, group=split))+
  geom_point(size=0.35)+
  geom_errorbar(aes(ymin = y_unc_mean-y_unc_se, ymax = y_unc_mean+y_unc_se), width=0.25, size=0.35) +
  geom_line(size=0.35, alpha=0.5)+
  # coord_cartesian(ylim=c(0.238, 0.34))+
  labs(y='Uncertainty', x='Unfamiliarity bin') +
  scale_fill_manual(values = c('#97a4ab','#efc57b')) + 
  scale_color_manual(values = c('#97a4ab','#efc57b')) + 
  default_theme + theme(#legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"))


# save to pdf
pdf('plots/figures/sfig5.pdf', width = 70/25.4, height = 50/25.4)
print(sfig5)
dev.off()
