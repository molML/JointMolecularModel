# This file plots the main results (fig4) of the paper
#
# Derek van Tilborg
# Eindhoven University of Technology
# Juli 2025
# loading some libraries

library(readr)
library(drc)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggrepel)
library(cowplot)
library(patchwork)

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
  # legend.position = 'none',
  legend.title = element_text(size=8),
  legend.background = element_blank(),
  legend.text = element_text(size=8),
  legend.spacing.y = unit(0., 'cm'),
  legend.key.size = unit(0.25, 'cm'),
  legend.key.width = unit(0.5, 'cm'),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank())


setwd("~/Dropbox/PycharmProjects/JointMolecularModel")

# Load the data
df_4_ad <- read_csv('plots/data/df_4_ad.csv')
df_4_bcefg <- read_csv('plots/data/df_4_bcefg.csv')


df_heatmap_pim = subset(df_4_bcefg, Type == 'Screen' & Protein == 'PIM1')
df_heatmap_cdk = subset(df_4_bcefg, Type == 'Screen' & Protein == 'CDK1')

col_a = '#b75a33'
col_b = '#577788'
col_c = '#efc57b'

#### PIM1 ####

fig4a = ggplot(subset(df_4_ad, Protein %in% c('PIM1')),
               aes(x=unfamiliarity, y=y_unc, fill=Method))+
  geom_point(size=1.5, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  scale_fill_manual(values=c(col_b, col_c, col_a))+
  labs(x='U(x)', y='H(x)')+
  default_theme + theme(legend.position = 'none')

fig4b = ggplot(df_heatmap_pim, aes(x = Tanimoto_to_dataset_max, y = Activity, fill=method_ABC))+
  geom_point(size=1.5, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  geom_text(aes(label = label_scatter), size=2, nudge_x = 0.01, nudge_y = 0.01, color = "black") +
  scale_y_continuous(limit=c(0, 120), breaks=c(0,25,50,75,100, 120)) +
  scale_x_continuous(limits=c(0.15, 0.45)) +
  labs(y='PIM1 activity (%)', x='Max similarity to training data', title='') +
  scale_fill_manual(values=c(col_a, col_b, col_c))+
  default_theme + theme(legend.position = 'none')

fig4c = ggplot(df_heatmap_pim, aes(x = method_ABC, y = Activity, fill=method_ABC))+
  geom_jitter(aes(fill=method_ABC), position=position_jitterdodge(0), size=1, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.5, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_y_continuous(limit=c(0, 120), breaks=c(0,25,50,75,100, 120)) +
  labs(x='Method', title='', y='PIM1 activity (%)') +
  geom_hline(yintercept=100, linewidth = 0.5, linetype = "solid") +
  geom_hline(yintercept=min(df_heatmap_pim$Activity), linewidth = 0.5, linetype = "dashed") +
  scale_fill_manual(values=c(col_a, col_b, col_c))+
  default_theme + theme(legend.position = 'none',
                        axis.title.y=element_blank(),
                        plot.margin = margin(t = 0,  # Top margin
                                             # r = 0,  # Right margin
                                             b = 5,  # Bottom margin
                                             l = 0)) # trbl

wilcox.test(subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'A')$Activity,
            subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'B')$Activity, 
            paired=F, alternative = 'two.sided')

wilcox.test(subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'B')$Activity,
            subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'C')$Activity, 
            paired=F, alternative = 'two.sided')

wilcox.test(subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'A')$Activity,
            subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'C')$Activity, 
            paired=F, alternative = 'two.sided')


#### CDK1 ####

fig4d = ggplot(subset(screening_lookup_table, Protein %in% c('CDK1')),
               aes(x=unfamiliarity, y=y_unc, fill=Method))+
  geom_point(size=1.5, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  scale_fill_manual(values=c(col_b, col_c, col_a))+
  labs(x='U(x)', y='H(x)')+
  default_theme + theme(legend.position = 'none')

# CDK1 activity plot
fig4e = ggplot(df_heatmap_cdk, aes(x = Tanimoto_to_dataset_max, y = Activity, fill=method_ABC))+
  geom_point(size=1.5, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  geom_text(aes(label = label_scatter), size=2, nudge_x = 0.01, nudge_y = 0.01, color = "black") +
  scale_y_continuous(limit=c(0, 120), breaks=c(0,25,50,75,100, 120)) +
  scale_x_continuous(limits=c(0.15, 0.45)) +
  labs(y='CDK1 activity (%)', x='Max similarity to training data', title='') +
  scale_fill_manual(values=c(col_a, col_b, col_c))+
  default_theme + theme(legend.position = 'none')

fig4f = ggplot(df_heatmap_cdk, aes(x = method_ABC, y = Activity, fill=method_ABC))+
  geom_jitter(aes(fill=method_ABC), position=position_jitterdodge(0), size=1, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.5, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_y_continuous(limit=c(0, 120), breaks=c(0,25,50,75,100, 120)) +
  labs(x='Method', title='', y='CDK1 activity (%)') +
  geom_hline(yintercept=100, linewidth = 0.5, linetype = "solid") +
  geom_hline(yintercept=min(df_heatmap_pim$Activity), linewidth = 0.5, linetype = "dashed") +
  scale_fill_manual(values=c(col_a, col_b, col_c))+
  default_theme + theme(legend.position = 'none',
                        axis.title.y=element_blank(),
                        plot.margin = margin(t = 0,  # Top margin
                                             # r = 0,  # Right margin
                                             b = 5,  # Bottom margin
                                             l = 0))

wilcox.test(subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'A')$Tanimoto_to_dataset_max,
            subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'B')$Tanimoto_to_dataset_max, 
            paired=F, alternative = 'two.sided')

wilcox.test(subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'B')$Tanimoto_to_dataset_max,
            subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'C')$Tanimoto_to_dataset_max, 
            paired=F, alternative = 'two.sided')

wilcox.test(subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'A')$Tanimoto_to_dataset_max,
            subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'C')$Tanimoto_to_dataset_max, 
            paired=F, alternative = 'two.sided')


df_heatmap_pim$label = paste0(round(df_heatmap_pim$Activity, 0), '%')
fig4g1 = ggplot(df_heatmap_pim, aes(x = Compound, y = Protein, fill = Activity)) +
  geom_tile() +
  scale_fill_gradient2(midpoint=100, low = "#4E7665", mid = "white", high = "#bbbbbb", name = "Activity")+
  geom_text(aes(label = label), color = "#101e25", size = 2) +
  default_theme + theme(legend.position = 'none',
                        axis.title.x=element_blank(),
                        axis.title.y=element_blank(),
                        axis.ticks.y=element_blank())

df_heatmap_cdk$label = paste0(round(df_heatmap_cdk$Activity, 0), '%')
fig4g2 = ggplot(df_heatmap_cdk, aes(x = Compound, y = Protein, fill = Activity)) +
  geom_tile() +
  scale_fill_gradient2(midpoint=100, low = "#4E7665", mid = "white", high = "#bbbbbb", name = "Activity")+
  geom_text(aes(label = label), color = "#101e25", size = 2) +
  default_theme + theme(legend.position = 'none',
                        axis.title.x=element_blank(),
                        axis.title.y=element_blank(),
                        axis.ticks.y=element_blank())

legend_plot = ggplot(df_heatmap, aes(x = Compound, y = Protein, fill = Activity)) +
  geom_tile() +
  scale_fill_gradient2(midpoint=100, low = "#4E7665", mid = "white", high = "#bbbbbb",
                       breaks = c(0, 25, 50, 75, 100), name = "Activity")

legend <- cowplot::get_legend(legend_plot)
ggdraw(legend)


fig_4g = plot_grid(fig4g1, fig4g2, ncol=1, align = "hv", axis = "tblr",
                   rel_heights = c(1, 1), labels = c('e', ''),
                   label_size=10)

fig_4g = plot_grid(fig_4g, legend, ncol=1, rel_heights = c(1, 0.075))

fig_4abcdef = plot_grid(fig4a, fig4b, fig4c, fig4d, fig4e, fig4f, ncol=6, align = "h", axis = "ltb",
                        rel_widths = c(0.9, 0.9, 0.3, 0.9, 0.9, 0.3), labels = c('a', 'b', 'c', 'd', 'e', 'f'),
                        label_size=10)

fig_4 = plot_grid(fig_4abcdef, fig_4g, rel_heights = c(1, 0.72), ncol=1)
fig_4


pdf('plots/figures/fig4.pdf', width = 180/25.4, height = 70/25.4)
print(fig_4)
dev.off()
