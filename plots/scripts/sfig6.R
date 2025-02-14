# This file plots the supplementary figure sFig6 of the paper
#
# Derek van Tilborg
# Eindhoven University of Technology
# February 2025

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


# the color scheme I used throughout the paper
cols = c('#577788','#97a4ab','#ef9d43','#efc57b', '#578d88', '#99beae')

data_to_bi_pca = function(df, axes = c(1, 2), loading_scaling=0.7, scale=T){
  #' Create a dataframe of eigenvectors and loadings from some data
  
  pca <- prcomp(df, scale = scale)
  
  scree = fviz_eig(pca)
  var <- facto_summarize(pca, element = "var", 
                         result = c("coord", "contrib", 'cos2'), 
                         axes = axes)
  ind <- facto_summarize(pca, element = "ind", 
                         result = c("coord", "contrib", 'cos2'), 
                         axes = axes)
  
  colnames(var)[2:3] <- c("x", "y")
  colnames(ind)[2:3] <- c("x", "y")
  
  # Scale Loadings 
  r <- min((max(ind[, "x"]) - min(ind[, "x"])/(max(var[, "x"]) - 
                                                 min(var[, "x"]))), (max(ind[, "y"]) - min(ind[, "y"])/
                                                                       (max(var[, "y"]) - min(var[, "y"]))))
  var[, c("x", "y")] <- var[, c("x", "y")] * r * loading_scaling
  
  # Merge the indivduals (eigenvectors) and variables 
  # (loading rotations, now scaled)
  var$type = rep('Loading', nrow(var))
  ind$type = rep('Score', nrow(ind))
  bi = rbind(ind, var)
  bi$cos2[bi$type == 'Score'] = 0
  bi$contrib[bi$type == 'Score'] = 0
  
  return( list('bi'=bi, 'pca'=pca, 'scree'=scree) )
}


data_to_biplot = function(pca_dat, val_var="MCSF", loading_scaling=1.6, lower_better=TRUE){
  #' convert a dataframe into pca data + best/worst scaling
  
  M_all = dcast(data = pca_dat, formula = ranking_method~dataset, value.var = val_var)
  
  rownames(M_all) = str_to_sentence(gsub('_', ' ', unlist(M_all[1])))
  M_all = M_all[2:ncol(M_all)]
  M_all= data.frame(t(M_all))
  
  if (lower_better == TRUE){
    M_all$Best = apply(M_all, 1, FUN = min)
    M_all$Worst = apply(M_all, 1, FUN = max)
  } else {
    M_all$Best = apply(M_all, 1, FUN = max)
    M_all$Worst = apply(M_all, 1, FUN = min)
  }
  
  colnames(M_all) = gsub('9', ' - ', gsub('\\.', ' ', colnames(M_all)))
  
  
  pca_all = data_to_bi_pca(t(M_all), loading_scaling=loading_scaling, scale=F)
  scree_all = pca_all$scree
  
  pca_all$bi$name = gsub('Best', 'Best - Best', pca_all$bi$name)
  pca_all$bi$name = gsub('Worst', 'Worst - Worst', pca_all$bi$name)
  
  return(pca_all)
  
}



# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointChemicalModel")


df_s5_abc <- read_csv('plots/data/df_4.csv')

df_s5 <- read_csv('results/screening_mols_properties_top50.csv')
df_s5 = subset(df_s5, ranking_method %in% c('utopia_dist_E_min_unc',
                                          'utopia_dist_E_min_ood',
                                          'utopia_dist_E_min_unc_max_ood',
                                          'utopia_dist_E_min_ood_min_cats',
                                          'utopia_dist_E_min_ood_min_mcsf',
                                          'utopia_dist_E_min_unc_min_cats',
                                          'utopia_dist_E_min_unc_min_mcsf'
))

df_s5$ranking_method = gsub('utopia_dist_E_min_unc_max_ood', 'Certain unfamiliar', df_s5$ranking_method)
df_s5$ranking_method = gsub('utopia_dist_E_min_ood_min_cats', 'Least unfamiliar novel cats', df_s5$ranking_method)
df_s5$ranking_method = gsub('utopia_dist_E_min_ood_min_mcsf', 'Least unfamiliar novel cores', df_s5$ranking_method)
df_s5$ranking_method = gsub('utopia_dist_E_min_unc_min_cats', 'Least uncertain novel cats', df_s5$ranking_method)
df_s5$ranking_method = gsub('utopia_dist_E_min_unc_min_mcsf', 'Least uncertain novel cores', df_s5$ranking_method)
df_s5$ranking_method = gsub('utopia_dist_E_min_unc', 'Least uncertain', df_s5$ranking_method)
df_s5$ranking_method = gsub('utopia_dist_E_min_ood', 'Least unfamiliar', df_s5$ranking_method)


df_s5$ranking_method = factor(df_s5$ranking_method, levels = rev(c(
  'Least uncertain', 
  'Least unfamiliar', 
  'Certain unfamiliar',
  'Least uncertain novel cores',   
  'Least unfamiliar novel cores',
  'Least uncertain novel cats',
  'Least unfamiliar novel cats'
)))


descr_cols = list(cols = c('#efc57b','#ef9d43','#b75a33',
                           '#97a4ab', '#577788',
                           '#99beae','#578d88', 
                           '#ffffff', '#101e25', '#101e25'),
                  descr =  c("Least uncertain", "Least unfamiliar", "Certain unfamiliar", 
                             "Least uncertain novel cores", "Least unfamiliar novel cores",
                             "Least uncertain novel cats", "Least unfamiliar novel cats", 
                             "", 'Best', 'Worst'))


#### sFig6 a, b - boxplots ####

sfig6a = ggplot(df_s5, aes(y = `Hit rate`, x = ranking_method, fill=ranking_method))+
  labs(y='Hit rate', x='', title='Higher = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_s5$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_s5$ranking_method), descr_cols[2][[1]])]) +
  scale_y_continuous(limit=c(0.25, 1)) +
  coord_flip() +
  default_theme + theme(legend.position = 'none',
                        axis.text.y=element_blank(),
                        plot.margin = unit(c(0.5, 2, 0.2, 0.2), "cm"))

mean(subset(df_s5, ranking_method == 'Least uncertain')$`Hit rate`)
sd(subset(df_s5, ranking_method == 'Least uncertain')$`Hit rate`)

mean(subset(df_s5, ranking_method == 'Least unfamiliar')$`Hit rate`)
sd(subset(df_s5, ranking_method == 'Least unfamiliar')$`Hit rate`)

# Wilcoxon signed-rank test
print('fig s5a Wilcoxon signed-rank tests:')
for (method_ in unique(df_s5$ranking_method)){
  
  wx = wilcox.test(subset(df_s5, ranking_method == method_)$`Hit rate`,
                   subset(df_s5, ranking_method == 'Least uncertain')$`Hit rate`,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0('Least uncertain', ' + ', method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}

wx = wilcox.test(subset(df_s5, ranking_method == 'Least unfamiliar')$`Hit rate`, subset(df_s5, ranking_method == 'Certain unfamiliar')$`Hit rate`, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar + Certain unfamiliar', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_s5, ranking_method == 'Least unfamiliar novel cats')$`Hit rate`, subset(df_s5, ranking_method == 'Least uncertain novel cats')$`Hit rate`, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cats + Least uncertain novel cats', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_s5, ranking_method == 'Least unfamiliar novel cores')$`Hit rate`, subset(df_s5, ranking_method == 'Least uncertain novel cores')$`Hit rate`, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cores + Least uncertain novel cores', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))



sfig6b = ggplot(df_s5, aes(y = Enrichment, x = ranking_method, fill=ranking_method))+
  labs(y='Enrichment', x='', title='Higher = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_s5$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_s5$ranking_method), descr_cols[2][[1]])]) +
  # scale_y_continuous(limit=c(0.125, 0.5)) +
  coord_flip() +
  default_theme + theme(legend.position = 'none',
                        axis.text.y=element_blank(),
                        plot.margin = unit(c(0.5, 2, 0.2, 0.2), "cm"))


# Wilcoxon signed-rank test
print('fig s5b Wilcoxon signed-rank tests:')
for (method_ in unique(df_s5$ranking_method)){
  
  wx = wilcox.test(subset(df_s5, ranking_method == method_)$Enrichment,
                   subset(df_s5, ranking_method == 'Least uncertain')$Enrichment,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0('Least uncertain', ' + ', method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}

wx = wilcox.test(subset(df_s5, ranking_method == 'Least unfamiliar')$Enrichment, subset(df_s5, ranking_method == 'Certain unfamiliar')$Enrichment, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar + Certain unfamiliar', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_s5, ranking_method == 'Least unfamiliar novel cats')$Enrichment, subset(df_s5, ranking_method == 'Least uncertain novel cats')$Enrichment, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cats + Least uncertain novel cats', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_s5, ranking_method == 'Least unfamiliar novel cores')$Enrichment, subset(df_s5, ranking_method == 'Least uncertain novel cores')$Enrichment, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cores + Least uncertain novel cores', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))




#### sFig6 c, d - PCA #####

# PCA of precision
c_pca_all = data_to_biplot(df_s5, val_var="Hit rate", lower_better = F)
c_bi_all = c_pca_all$bi

# Get xy the coordinates for the best and worst points
c_best = unlist(subset(c_bi_all, name == 'Best - Best')[c(2,3)])
c_worst = unlist(subset(c_bi_all, name == 'Worst - Worst')[c(2,3)])

# invert order of axis
# c_bi_all$x = -1 * c_bi_all$x

c_x_axis_label = paste0('PC1 (',round(c_pca_all$scree$data$eig[1],1),'%)')
c_y_axis_label = paste0('PC2 (',round(c_pca_all$scree$data$eig[2],1),'%)')

c_bi_all$name = gsub('Best - ', '', c_bi_all$name)
c_bi_all$name = gsub('Worst - ', '', c_bi_all$name)

c_bi_all$cols = c_bi_all$name
c_bi_all$cols[!c_bi_all$cols %in% descr_cols$descr] = ''
c_bi_all$cols = factor(c_bi_all$cols, levels=c_bi_all$cols[1:9])

# Make the actual plot
sfig6c = ggplot(c_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(c_bi_all$type == 'Score', 0.8, 0), color = "#101e25", stroke=0.1) +
  geom_text(aes(label = name), alpha = ifelse(c_bi_all$type == 'Score', 1, 0),
            size = 2, segment.size = 0.25, force = 30, max.iter = 1505,
            max.overlaps = 200, show.legend = FALSE) +
  labs(x = c_x_axis_label, y = c_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(c_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-0.3, 0.3))+
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.25, 0.5, 0.25, 0.5), "cm"))


# PCA of MCSF
d_pca_all = data_to_biplot(df_s5, val_var="Enrichment", lower_better = T)
d_bi_all = d_pca_all$bi

# Get xy the coordinates for the best and worst points
d_best = unlist(subset(d_bi_all, name == 'Best - Best')[c(2,3)])
d_worst = unlist(subset(d_bi_all, name == 'Worst - Worst')[c(2,3)])

# invert order of axis
# d_bi_all$x = -1 * d_bi_all$x

d_x_axis_label = paste0('PC1 (',round(d_pca_all$scree$data$eig[1],1),'%)')
d_y_axis_label = paste0('PC2 (',round(d_pca_all$scree$data$eig[2],1),'%)')

d_bi_all$name = gsub('Best - ', '', d_bi_all$name)
d_bi_all$name = gsub('Worst - ', '', d_bi_all$name)

d_bi_all$cols = d_bi_all$name
d_bi_all$cols[!d_bi_all$cols %in% descr_cols$descr] = ''
d_bi_all$cols = factor(d_bi_all$cols, levels=d_bi_all$cols[1:9])

# Make the actual plot
sfig6d = ggplot(d_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(d_bi_all$type == 'Score', 0.8, 0), color = "#101e25", stroke=0.1) +
  geom_text(aes(label = name), alpha = ifelse(d_bi_all$type == 'Score', 1, 0),
            size = 2, segment.size = 0.25, force = 50, max.iter = 1505,
            max.overlaps = 200, show.legend = FALSE) +
  labs(x = d_x_axis_label, y = d_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(d_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-1, 1))+
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.25, 0.5, 0.25, 0.5), "cm"))


#### sfig6 #####



sfig6ab = plot_grid(plot_spacer(), sfig6a, sfig6b,
                    rel_widths = c(0.5, 1, 1),
                    labels = c('a', '', 'b'), label_size = 10, ncol=3)

sfig6cd = plot_grid(sfig6c, sfig6d,
                    rel_widths = c(1, 1),
                    labels = c('c', 'd'), label_size = 10, ncol=2)

sfig6 = plot_grid(sfig6ab, sfig6cd,
                 rel_heights = c(1.2, 1),
                 ncol=1)

sfig6

# save to pdf
pdf('plots/figures/sfig6.pdf', width = 120/25.4, height = 90/25.4)
print(sfig6)
dev.off()
