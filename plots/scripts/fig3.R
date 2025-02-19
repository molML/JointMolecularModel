# This file plots the main results (fig3) of the paper
#
# Derek van Tilborg
# Eindhoven University of Technology
# January 2025

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

descr_cols = list(cols = c('#efc57b','#ef9d43','#b75a33',
                           '#97a4ab', '#577788',
                           '#99beae','#578d88', 
                           '#ffffff', '#101e25', '#101e25'),
                  descr =  c("Least uncertain", "Least unfamiliar", "Certain unfamiliar", 
                             "Least uncertain novel cores", "Least unfamiliar novel cores",
                             "Least uncertain novel cats", "Least unfamiliar novel cats", 
                             "", 'Best', 'Worst'))


# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointChemicalModel")

df_3_abc <- read_csv('plots/data/df_3.csv')
df_3_efg <- read_csv('plots/data/df_3efg.csv')

# Only keep the relevant ranking methods
df_3_efg = subset(df_3_efg, ranking_method %in% c('utopia_dist_E_min_unc',
                                          'utopia_dist_E_min_ood',
                                          'utopia_dist_E_min_unc_max_ood',

                                          'utopia_dist_E_min_ood_min_cats',
                                          'utopia_dist_E_min_ood_min_mcsf',
                                          
                                          'utopia_dist_E_min_unc_min_cats',
                                          'utopia_dist_E_min_unc_min_mcsf'
                                          ))

df_3_efg$ranking_method = gsub('utopia_dist_E_min_unc_max_ood', 'Certain unfamiliar', df_3_efg$ranking_method)
df_3_efg$ranking_method = gsub('utopia_dist_E_min_ood_min_cats', 'Least unfamiliar novel cats', df_3_efg$ranking_method)
df_3_efg$ranking_method = gsub('utopia_dist_E_min_ood_min_mcsf', 'Least unfamiliar novel cores', df_3_efg$ranking_method)
df_3_efg$ranking_method = gsub('utopia_dist_E_min_unc_min_cats', 'Least uncertain novel cats', df_3_efg$ranking_method)
df_3_efg$ranking_method = gsub('utopia_dist_E_min_unc_min_mcsf', 'Least uncertain novel cores', df_3_efg$ranking_method)
df_3_efg$ranking_method = gsub('utopia_dist_E_min_unc', 'Least uncertain', df_3_efg$ranking_method)
df_3_efg$ranking_method = gsub('utopia_dist_E_min_ood', 'Least unfamiliar', df_3_efg$ranking_method)

# Order factors
df_3_efg$ranking_method = factor(df_3_efg$ranking_method, levels = rev(c(
                                                             'Least uncertain', 
                                                             'Least unfamiliar', 
                                                             'Certain unfamiliar',
                                                             'Least uncertain novel cores',   
                                                             'Least unfamiliar novel cores',
                                                             'Least uncertain novel cats',
                                                             'Least unfamiliar novel cats'
                                                             )))



####  Fig3 a, b - scatter plots ####

dataset_name_ = 'CHEMBL262_Ki'

df_3a = subset(df_3_abc, dataset == dataset_name_ & ranking_method == 'utopia_dist_E_min_unc')
df_3a$topk_member = as.character(rank(df_3a$utopia_dist, ties.method = "min") <= 50)
fig3a = ggplot(df_3a, aes(x = y_E_, y = y_unc, fill = topk_member)) +
  geom_point(size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  scale_fill_manual(values = c('#dddddd', '#efc57b')) +
  labs(x='Expected value', y='Uncertainty')+
  default_theme + theme(legend.position = 'none')


df_3b = subset(df_3_abc, dataset == dataset_name_ & ranking_method == 'utopia_dist_E_min_ood')
df_3b$topk_member = as.character(rank(df_3b$utopia_dist, ties.method = "min") <= 50)
fig3b = ggplot(df_3b, aes(x = y_E_, y = ood_score, fill = topk_member)) +
  geom_point(size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  scale_fill_manual(values = c('#dddddd', '#ef9d43')) +
  labs(x='Expected value', y='Unfamiliarity')+
  default_theme + theme(legend.position = 'none')


####  Fig3 c, d - 3D plots ####

fig3c = plot_spacer()
fig3d = plot_spacer()


pdf('plots/figures/fig3cd.pdf', width = 8, height = 4)
layout(matrix(1:2, nrow = 1, byrow = TRUE))  # Arrange two plots side by side

scaled_values = 0.5 + (df_3c$ood_score-min(df_3c$ood_score))*(0.5/(max(df_3c$ood_score)-min(df_3c$ood_score)))
alpha_values = 225 * scaled_values
colors = c()
for (i in 1:length(df_3c$topk_member)){
  if (df_3c$topk_member[i] == 'TRUE'){
    colors = c(colors, rgb(col2rgb('#b75a33')[1], col2rgb('#b75a33')[2], col2rgb('#b75a33')[3], maxColorValue = 255, alpha=alpha_values[i]))
  } else {
    colors = c(colors, rgb(200, 200, 200, maxColorValue = 255, alpha=alpha_values[i]))
  }
}

scatterplot3d(df_3c$ood_score, df_3c$y_unc, df_3c$y_E_, grid = TRUE, box = TRUE, asp = 0.7, xlim=c(-0.1, 1.6), lab=c(6, 3),  ylim=c(0.8, 0),
              pch = 20, type = "p", main = "3-way selection", angle = -135, color = colors)


scaled_values = 0.5 + (df_3d$MCSF_-min(df_3d$MCSF_))*(0.5/(max(df_3d$MCSF_)-min(df_3d$MCSF_)))
alpha_values = 225 * scaled_values
colors = c()
for (i in 1:length(df_3d$topk_member)){
  if (df_3d$topk_member[i] == 'TRUE'){
    colors = c(colors, rgb(col2rgb('#577788')[1], col2rgb('#577788')[2], col2rgb('#577788')[3], maxColorValue = 255, alpha=alpha_values[i]))
  } else {
    colors = c(colors, rgb(200, 200, 200, maxColorValue = 255, alpha=alpha_values[i]))
  }
}

scatterplot3d(-df_3d$MCSF_, df_3d$ood_score, df_3d$y_E_, grid = TRUE, box = TRUE, axis = T, asp = 1.42, lab=c(6, 4), lab.z=c(6), ylim=c(-0.1, 1.3), xlim=c(-0.5, -0),
              pch = 20, type = "p", main = "3-way selection", angle = -135, color = colors)

dev.off()



#### Fig3 e, f, g - boxplots ####

fig3e = ggplot(df_3_efg, aes(y = Precision, x = ranking_method, fill=ranking_method))+
  labs(y='Precision\n', x='', title='Higher = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_3_efg$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_3_efg$ranking_method), descr_cols[2][[1]])]) +
  scale_y_continuous(limit=c(0.25, 1)) +
  coord_flip() +
  default_theme + theme(legend.position = 'none',
                        axis.text.y=element_blank(),
                        plot.margin = unit(c(0.5, 2, 0.2, 0.2), "cm"))

print(paste0('Least uncertain precision: ', 
             round(mean(subset(df_3_efg, ranking_method == 'Least uncertain')$Precision),2),
             '±',
             round(se(subset(df_3_efg, ranking_method == 'Least uncertain')$Precision),2)))

print(paste0('Least unfamiliar precision: ', 
             round(mean(subset(df_3_efg, ranking_method == 'Least unfamiliar')$Precision),2),
             '±',
             round(se(subset(df_3_efg, ranking_method == 'Least unfamiliar')$Precision),2)))


# Wilcoxon signed-rank test
print('fig3e Wilcoxon signed-rank tests:')
for (method_ in unique(df_3_efg$ranking_method)){
  
  wx = wilcox.test(subset(df_3_efg, ranking_method == method_)$Precision,
                   subset(df_3_efg, ranking_method == 'Least uncertain')$Precision,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0('Least uncertain', ' + ', method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar')$Precision, subset(df_3_efg, ranking_method == 'Certain unfamiliar')$Precision, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar + Certain unfamiliar', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar novel cats')$Precision, subset(df_3_efg, ranking_method == 'Least uncertain novel cats')$Precision, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cats + Least uncertain novel cats', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar novel cores')$Precision, subset(df_3_efg, ranking_method == 'Least uncertain novel cores')$Precision, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cores + Least uncertain novel cores', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))



fig3f = ggplot(df_3_efg, aes(y = MCSF_hits_to_train_actives, x = ranking_method, fill=ranking_method))+
    labs(y='Mol. core overlap\nto known actives', x='', title='Lower = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_3_efg$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_3_efg$ranking_method), descr_cols[2][[1]])]) +
  scale_y_continuous(limit=c(0.125, 0.5)) +
  coord_flip() +
  default_theme + theme(legend.position = 'none',
                        axis.text.y=element_blank(),
                        plot.margin = unit(c(0.5, 2, 0.2, 0.2), "cm"))


# Wilcoxon signed-rank test
print('fig3f Wilcoxon signed-rank tests:')
for (method_ in unique(df_3_efg$ranking_method)){
  
  wx = wilcox.test(subset(df_3_efg, ranking_method == method_)$MCSF_hits_to_train_actives,
                   subset(df_3_efg, ranking_method == 'Least uncertain')$MCSF_hits_to_train_actives,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0('Least uncertain', ' + ', method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar')$MCSF_hits_to_train_actives, subset(df_3_efg, ranking_method == 'Certain unfamiliar')$MCSF_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar + Certain unfamiliar', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar novel cats')$MCSF_hits_to_train_actives, subset(df_3_efg, ranking_method == 'Least uncertain novel cats')$MCSF_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cats + Least uncertain novel cats', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar novel cores')$MCSF_hits_to_train_actives, subset(df_3_efg, ranking_method == 'Least uncertain novel cores')$MCSF_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cores + Least uncertain novel cores', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))




fig3g = ggplot(df_3_efg, aes(y = pharmacophore_hits_to_train_actives, x = ranking_method, fill=ranking_method))+
  labs(y='Pharmacophore sim.\nto known actives', x='', title='Lower = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_3_efg$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_3_efg$ranking_method), descr_cols[2][[1]])]) +
  scale_y_continuous(limit=c(0.1, 0.65)) +
  coord_flip() +
  default_theme + theme(legend.position = 'none',
                        axis.text.y=element_blank(),
                        plot.margin = unit(c(0.5, 2, 0.2, 0.2), "cm"))


# Wilcoxon signed-rank test
print('fig3g Wilcoxon signed-rank tests:')
for (method_ in unique(df_3_efg$ranking_method)){
  
  wx = wilcox.test(subset(df_3_efg, ranking_method == method_)$pharmacophore_hits_to_train_actives,
                   subset(df_3_efg, ranking_method == 'Least uncertain')$pharmacophore_hits_to_train_actives,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0('Least uncertain', ' + ', method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar')$pharmacophore_hits_to_train_actives, subset(df_3_efg, ranking_method == 'Certain unfamiliar')$pharmacophore_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar + Certain unfamiliar', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar novel cats')$pharmacophore_hits_to_train_actives, subset(df_3_efg, ranking_method == 'Least uncertain novel cats')$pharmacophore_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cats + Least uncertain novel cats', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))

wx = wilcox.test(subset(df_3_efg, ranking_method == 'Least unfamiliar novel cores')$pharmacophore_hits_to_train_actives, subset(df_3_efg, ranking_method == 'Least uncertain novel cores')$pharmacophore_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')
print(paste0('Least unfamiliar novel cores + Least uncertain novel cores', ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))



#### Fig3 h, i, j - PCA #####

# PCA of precision
h_pca_all = data_to_biplot(df_3_efg, val_var="Precision", lower_better = F)
h_bi_all = h_pca_all$bi

# Get xy the coordinates for the best and worst points
h_best = unlist(subset(h_bi_all, name == 'Best - Best')[c(2,3)])
h_worst = unlist(subset(h_bi_all, name == 'Worst - Worst')[c(2,3)])

# invert order of axis
# h_bi_all$x = -1 * h_bi_all$x

h_x_axis_label = paste0('PC1 (',round(h_pca_all$scree$data$eig[1],1),'%)')
h_y_axis_label = paste0('PC2 (',round(h_pca_all$scree$data$eig[2],1),'%)')

h_bi_all$name = gsub('Best - ', '', h_bi_all$name)
h_bi_all$name = gsub('Worst - ', '', h_bi_all$name)

h_bi_all$cols = h_bi_all$name
h_bi_all$cols[!h_bi_all$cols %in% descr_cols$descr] = ''
h_bi_all$cols = factor(h_bi_all$cols, levels=h_bi_all$cols[1:9])

# Make the actual plot
fig3h = ggplot(h_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(h_bi_all$type == 'Score', 0.8, 0), color = "#101e25", stroke=0.1) +
  geom_text(aes(label = name), alpha = ifelse(h_bi_all$type == 'Score', 1, 0),
            size = 2, segment.size = 0.25, force = 30, max.iter = 1505,
            max.overlaps = 200, show.legend = FALSE) +
  labs(x = h_x_axis_label, y = h_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(h_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-0.6, 0.6))+
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.25, 0.5, 0.25, 0.5), "cm"))


# PCA of MCSF
i_pca_all = data_to_biplot(df_3_efg, val_var="MCSF_hits_to_train_actives", lower_better = T)
i_bi_all = i_pca_all$bi

# Get xy the coordinates for the best and worst points
i_best = unlist(subset(i_bi_all, name == 'Best - Best')[c(2,3)])
i_worst = unlist(subset(i_bi_all, name == 'Worst - Worst')[c(2,3)])

# invert order of axis
i_bi_all$x = -1 * i_bi_all$x

i_x_axis_label = paste0('PC1 (',round(i_pca_all$scree$data$eig[1],1),'%)')
i_y_axis_label = paste0('PC2 (',round(i_pca_all$scree$data$eig[2],1),'%)')

i_bi_all$name = gsub('Best - ', '', i_bi_all$name)
i_bi_all$name = gsub('Worst - ', '', i_bi_all$name)

i_bi_all$cols = i_bi_all$name
i_bi_all$cols[!i_bi_all$cols %in% descr_cols$descr] = ''
i_bi_all$cols = factor(i_bi_all$cols, levels=i_bi_all$cols[1:9])

# Make the actual plot
fig3i = ggplot(i_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(i_bi_all$type == 'Score', 0.8, 0), color = "#101e25", stroke=0.1) +
  geom_text(aes(label = name), alpha = ifelse(i_bi_all$type == 'Score', 1, 0),
            size = 2, segment.size = 0.25, force = 50, max.iter = 1505,
            max.overlaps = 200, show.legend = FALSE) +
  labs(x = i_x_axis_label, y = i_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(i_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-0.15, 0.15))+
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.25, 0.5, 0.25, 0.5), "cm"))


# PCA of pharmacophore sim
j_pca_all = data_to_biplot(df_3_efg, val_var="pharmacophore_hits_to_train_actives", lower_better = F)
j_bi_all = j_pca_all$bi

# invert order of axis
# j_bi_all$x = -1 * j_bi_all$x

# Get xy the coordinates for the best and worst points
j_best = unlist(subset(j_bi_all, name == 'Best - Best')[c(2,3)])
j_worst = unlist(subset(j_bi_all, name == 'Worst - Worst')[c(2,3)])

j_x_axis_label = paste0('PC1 (',round(j_pca_all$scree$data$eig[1],1),'%)')
j_y_axis_label = paste0('PC2 (',round(j_pca_all$scree$data$eig[2],1),'%)')

j_bi_all$name = gsub('Best - ', '', j_bi_all$name)
j_bi_all$name = gsub('Worst - ', '', j_bi_all$name)

j_bi_all$cols = j_bi_all$name
j_bi_all$cols[!j_bi_all$cols %in% descr_cols$descr] = ''
j_bi_all$cols = factor(j_bi_all$cols, levels=j_bi_all$cols[1:9])

# Make the actual plot
fig3j = ggplot(j_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(j_bi_all$type == 'Score', 0.8, 0), color = "#101e25", stroke=0.1) +
  geom_text(aes(label = name), alpha = ifelse(j_bi_all$type == 'Score', 1, 0),
            size = 2, segment.size = 0.25, force = 30, max.iter = 1505,
            max.overlaps = 30, show.legend = FALSE) +
  labs(x = j_x_axis_label, y = j_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(j_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-0.1, 0.1))+
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.25, 0.5, 0.25, 0.5), "cm"))


#### Fig3 #####

fig3abcd =  plot_grid(fig3a, fig3b, fig3c, fig3d, 
                     labels = c('a', 'b', 'c', 'd'),
                     label_size = 10, ncol=4)

fig3efg = plot_grid(plot_spacer(), fig3e, fig3f, fig3g,
                    rel_widths = c(0.5, 1, 1, 1),
                    labels = c('e', '', 'f', 'g'), label_size = 10, ncol=4)

fig3hij = plot_grid(fig3h, fig3i, fig3j,
                    rel_widths = c(1, 1, 1),
                    labels = c('h', 'i', 'j'), label_size = 10, ncol=3)

fig3 = plot_grid(fig3abcd, fig3efg, fig3hij,
                 rel_heights = c(1, 1.2, 1),
                 ncol=1)

fig3
 
# save to pdf
pdf('plots/figures/fig3_.pdf', width = 180/25.4, height = 130/25.4)
print(fig3)
dev.off()
