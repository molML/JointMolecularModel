# This file plots the supplementary figure sFig1 of the paper
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


scatter_plot = function(df, dataset_name){
  
  dataset_title = gsub('_', ' ', dataset_name)
  if (grepl('EC50', dataset_name)){
    dataset_title = gsub('EC50', 'EC', gsub('CHEMBL', 'ChEMBL', dataset_title))
    print(dataset_title)
    dataset_title = bquote(.(dataset_title)[50])
  }
  
  if (grepl('Ki', dataset_name)){
    dataset_title = gsub('Ki', 'K', gsub('CHEMBL', 'ChEMBL', dataset_title))
    dataset_title = bquote(.(dataset_title)['i'])
  }
  
  p = ggplot(subset(df, dataset == dataset_name), aes(x=x, y=y, fill=Split, color=Split))+
    geom_point(alpha = 0.80, shape=21, size = 1, stroke = 0.1) +
    scale_fill_manual(values = c('#577788','#577788', '#577788')) +
    scale_color_manual(values = c('#577788','#577788', '#577788')) +
    labs(x='', y='', title=dataset_title) + 
    default_theme + theme(axis.ticks = element_blank(),
                          axis.text.x=element_blank(),
                          axis.text.y=element_blank(),
                          axis.title.x=element_blank(),
                          axis.title.y=element_blank(),
                          legend.position = 'none',
                          plot.margin = unit(c(0.08, 0.02, 0.02, 0.02), "cm"))
  
  return(p)
}


############

df <- read_csv("results/dataset_clustering/TSNE_coordinates.csv")

df = subset(df, fingerprint == 'scaffold')
df$Split = gsub('ood', 'OOD', df$Split)
df$Split = gsub('train', 'Train', df$Split)
df$Split = gsub('test', 'Test', df$Split)
df$Split = factor(df$Split, levels = c('OOD', 'Train', 'Test'))


all_plots = list()

for (dataset_name in unique(df$dataset)){
  all_plots[[length(all_plots) + 1]] = scatter_plot(df, dataset_name)
}


legend = get_legend(ggplot(df, aes(x=x, y=y, fill=Split, color=Split))+
                      geom_point() +
                      scale_color_manual(values = c('#577788','#577788', '#577788')) +
                      theme(legend.position = 'right', legend.key = element_blank()))

all_plots[[length(all_plots) + 1]] <- legend

plot_grid(plotlist = all_plots, ncol=6, scale = 1)

dev.print(pdf, 'plots/figures/sfig1.pdf', width = 180/25.4, height = 200/25.4) # width = 7.205, height = 4


# df_eig <- read_csv("results/Eigenvalues_ChEMBL4792_Ki.csv")
# 
# ggplot(df_eig, aes(x = index, y = eigenvalues))+
#   # geom_line(color = 'grey', size=0.25)+
#   labs(x = 'k') +
#   # geom_point(alpha = 0.80, shape=16, size = 1.5, color = '#577788') +
#   geom_point(alpha = 0.80, shape=21, size = 1.5, stroke=0.1, fill = '#97a4ab', color = '#577788') +
#   geom_vline(xintercept = 11, linetype='dashed', alpha=0.5, colour='#bfbab8', size=0.25)+
#   default_theme + theme(panel.border = element_blank(),
#                         axis.line.x.bottom=element_line(color="#101e25", size=0.35),
#                         axis.line.y.left=element_line(color="#101e25", size=0.35),)
# 
# dev.print(pdf, 'Fig_eigenvalues.pdf', width = 220/25.4/6, height = 200/25.4/6) # width = 7.205, height = 4
# 
# 
# df_clust <- read_csv("data/split/CHEMBL4792_Ki_split.csv")
# 
# df_clust_coord = subset(df, dataset == 'CHEMBL4792_Ki')
# df_clust_coord$cluster = factor(df_clust$cluster)
# 
# # clust_cols = c('#b75a33','#ef9d43','#efc57b','#f0dab3', 
# #                '#2b5a52','#578d88','#99beae','#bbc4ba',
# #                '#2c4653','#577788','#97a4ab','#bfbab8')
# 
# clust_cols = c('#ef9d43','#efc57b','#f0dab3', '#f4e7cb', 
#                '#578d88','#99beae','#bbc4ba', '#ccd3cb',
#                '#577788','#97a4ab','#bfbab8', '#d1ccc8')
# 
# ggplot(df_clust_coord, aes(x=x, y=y, fill=cluster, color=cluster))+
#   geom_point(alpha = 0.80, shape=21, size = 1, stroke = 0.1) +
#   scale_fill_manual(values = clust_cols) +
#   scale_color_manual(values = clust_cols) +
#   labs(x='', y='', title='') + 
#   default_theme + theme(axis.ticks = element_blank(),
#                         axis.text.x=element_blank(),
#                         axis.text.y=element_blank(),
#                         axis.title.x=element_blank(),
#                         axis.title.y=element_blank(),
#                         legend.position = 'none',
#                         plot.margin = unit(c(0.08, 0.02, 0.02, 0.02), "cm"))
# 
# dev.print(pdf, 'Fig_spectral_clust.pdf', width = 180/25.4/6, height = 200/25.4/6) # width = 7.205, height = 4

