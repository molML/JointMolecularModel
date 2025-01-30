# This file plots the main results (Fig4) of the paper
#
# Derek van Tilborg
# Eindhoven University of Technology
# January 2024

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


descr_cols = list(cols = c('#97a4ab','#efc57b','#b75a33','#ef9d43','#99beae', '#577788',
                           '#ffffff', '#101e25', '#101e25'),
                  descr =  c("Most uncertain", "Most unfamiliar", "Certain unfamiliar", "Least unfamiliar", "Best prediction", "Least uncertain", 
                             "", 'Best', 'Worst'))


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


df_4_abc <- read_csv('plots/data/df_4.csv')


df_4 <- read_csv('results/screening_mols_properties_top50.csv')
df_4 = subset(df_4, ranking_method %in% c('utopia_dist_E',
                                          'utopia_dist_E_min_unc',
                                          'utopia_dist_E_min_ood',
                                          'utopia_dist_E_max_unc',
                                          'utopia_dist_E_max_ood',
                                          'utopia_dist_E_min_unc_max_ood'))

df_4 = subset(df_4, dataset != 'ESR1_ant')

df_4$ranking_method = gsub('utopia_dist_E_min_unc_max_ood', 'Certain unfamiliar', df_4$ranking_method)
df_4$ranking_method = gsub('utopia_dist_E_min_unc', 'Least uncertain', df_4$ranking_method)
df_4$ranking_method = gsub('utopia_dist_E_min_ood', 'Least unfamiliar', df_4$ranking_method)
df_4$ranking_method = gsub('utopia_dist_E_max_unc', 'Most uncertain', df_4$ranking_method)
df_4$ranking_method = gsub('utopia_dist_E_max_ood', 'Most unfamiliar', df_4$ranking_method)
df_4$ranking_method = gsub('utopia_dist_E', 'Best prediction', df_4$ranking_method)

# df_4$ranking_method = factor(df_4$ranking_method, levels = names(sort(tapply(df_4$Precision, df_4$ranking_method, function(x) median(x, na.rm = TRUE)))))

df_4$ranking_method = factor(df_4$ranking_method, levels = c('Best prediction', 'Least uncertain', 'Most uncertain', 'Least unfamiliar', 'Most unfamiliar', 'Certain unfamiliar'))

# table(df_4$ranking_method)

######

dataset_name_ = 'CHEMBL262_Ki'

descr_cols$cols[match('Least unfamiliar', descr_cols$cols)]



df_4a = subset(df_4_abc, dataset == dataset_name_ & ranking_method == 'utopia_dist_E_min_ood')
df_4a$topk_member = as.character(rank(df_4a$utopia_dist, ties.method = "min") <= 50)

fig4a =ggplot(df_4a, aes(x = y_E_, y = ood_score, fill = topk_member)) +
  geom_point(size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  scale_fill_manual(values = c('#dddddd', '#ef9d43')) +
  labs(x='Expected value', y='Unfamiliarity')+
  default_theme + theme(legend.position = 'none')


df_4b = subset(df_4_abc, dataset == dataset_name_ & ranking_method == 'utopia_dist_E_min_unc')
df_4b$topk_member = as.character(rank(df_4b$utopia_dist, ties.method = "min") <= 50)

fig4b = ggplot(df_4b, aes(x = y_E_, y = y_unc, fill = topk_member)) +
  geom_point(size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  scale_fill_manual(values = c('#dddddd', '#577788')) +
  labs(x='Expected value', y='Uncertainty')+
  default_theme + theme(legend.position = 'none')

df_4c = subset(df_4_abc, dataset == dataset_name_ & ranking_method == 'utopia_dist_E_min_unc_max_ood')
df_4c$topk_member = as.character(rank(df_4c$utopia_dist, ties.method = "min") <= 50)

fig4c =ggplot(df_4c, aes(x = y_unc, y = ood_score, fill = topk_member)) +
  geom_point(aes(alpha=y_E_), size=1, shape=21, color = "black", stroke = 0.1) +
  scale_fill_manual(values = c('#dddddd', '#b75a33')) +
  labs(x='Uncertainty', y='Unfamiliarity')+
  default_theme + theme(legend.position = 'none')



scaled_values = 0.5 + (df_4c$ood_score-min(df_4c$ood_score))*(0.5/(max(df_4c$ood_score)-min(df_4c$ood_score)))
alpha_values = 225 * scaled_values
colors = c()
for (i in 1:length(df_4c$topk_member)){
  if (df_4c$topk_member[i] == 'TRUE'){
    colors = c(colors, rgb(col2rgb('#b75a33')[1], col2rgb('#b75a33')[2], col2rgb('#b75a33')[3], maxColorValue = 255, alpha=alpha_values[i]))
  } else {
    colors = c(colors, rgb(200, 200, 200, maxColorValue = 255, alpha=alpha_values[i]))
  }
}


pdf('plots/figures/fig4_3d.pdf', width = 4, height = 4)
scatterplot3d(df_4c$ood_score, df_4c$y_unc, df_4c$y_E_, grid = TRUE, box = TRUE, asp = 0.6,
              pch = 20, type = "p", main = "3-way selection", angle = -135, color = colors)
dev.off()




fig4d = ggplot(df_4, aes(y = Precision, x = ranking_method, fill=ranking_method))+
  labs(y='Precision', x='', title='Higher = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])]) +
  coord_cartesian(ylim=c(0.0, 1))+
  default_theme + theme(legend.position = 'none',
                        axis.text.x=element_blank(),
                        # axis.ticks.x=element_blank(),
                        plot.margin = unit(c(1, 0.2, 0, 0.2), "cm"))

# Wilcoxon signed-rank test
print('fig2d Wilcoxon signed-rank tests:')
for (method_ in unique(df_4$ranking_method)){
  
  wx = wilcox.test(subset(df_4, ranking_method == method_)$Precision,
                   subset(df_4, ranking_method == 'Best prediction')$Precision,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0(method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}


fig4e = ggplot(df_4, aes(y = Enrichment, x = ranking_method, fill=ranking_method))+
  labs(y='Enrichment', x='', title='Higher = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  coord_cartesian(ylim=c(-0.2, 3.4))+
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])]) +
  default_theme + theme(legend.position = 'none',
                        axis.text.x=element_blank(),
                        # axis.ticks.x=element_blank(),
                        plot.margin = unit(c(1, 0.2, 0, 0.2), "cm"))

# Wilcoxon signed-rank test
print('fig2d Wilcoxon signed-rank tests:')
for (method_ in unique(df_4$ranking_method)){
  
  wx = wilcox.test(subset(df_4, ranking_method == method_)$Enrichment,
                   subset(df_4, ranking_method == 'Best prediction')$Enrichment,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0(method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}


fig4f = ggplot(df_4, aes(y = MCSF_hits_to_train_actives, x = ranking_method, fill=ranking_method))+
  labs(y='Hit core overlap to known actives', x='', title='Lower = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])]) +
  coord_cartesian(ylim=c(0.15, 0.50))+
  default_theme + theme(legend.position = 'none',
                        axis.text.x=element_blank(),
                        # axis.ticks.x=element_blank(),
                        plot.margin = unit(c(1, 0.2, 0, 0.2), "cm"))

# Wilcoxon signed-rank test
print('fig2d Wilcoxon signed-rank tests:')
for (method_ in unique(df_4$ranking_method)){
  
  wx = wilcox.test(subset(df_4, ranking_method == method_)$MCSF_hits_to_train_actives,
                   subset(df_4, ranking_method == 'Best prediction')$MCSF_hits_to_train_actives,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0(method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}


fig4g = ggplot(df_4, aes(y = internal_tanimoto_hits, x = ranking_method, fill=ranking_method))+
  labs(y='Internal hit similarity', x='', title='Lower = better') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])]) +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])]) +
  coord_cartesian(ylim=c(0.05, 0.60))+
  default_theme + theme(legend.position = 'none',
                        axis.text.x=element_blank(),
                        # axis.ticks.x=element_blank(),
                        plot.margin = unit(c(1, 0.2, 0, 0.2), "cm"))

# Wilcoxon signed-rank test
print('fig2d Wilcoxon signed-rank tests:')
for (method_ in unique(df_4$ranking_method)){
  
  wx = wilcox.test(subset(df_4, ranking_method == method_)$internal_tanimoto_hits,
                   subset(df_4, ranking_method == 'Best prediction')$internal_tanimoto_hits,
                   paired=TRUE, alternative = 'two.sided')
  print(paste0(method_, ': ', ifelse(wx$p.value < 0.05, '*', 'n.s.'),' - ',  wx$p.value))
}

# 
# fig4 = plot_grid(fig4a, fig4b, fig4c, fig4d,
#                  # rel_widths = c(1, 1, 1, 1, 1),
#                  labels = c('a', 'b', 'c', 'd'), label_size = 10, ncol=4)
#   
# fig4

# save to pdf
# pdf('plots/figures/fig4.pdf', width = 180/25.4, height = 55/25.4)
# print(fig4)
# dev.off()


# Precision
# internal_tanimoto_hits
# Tanimoto_hits_to_train_actives
# unique_scaffolds_hits_ratio


##### PCA #####


h_pca_all = data_to_biplot(df_4, val_var="Precision", lower_better = F)
h_bi_all = h_pca_all$bi

# Get xy the coordinates for the best and worst points
h_best = unlist(subset(h_bi_all, name == 'Best - Best')[c(2,3)])
h_worst = unlist(subset(h_bi_all, name == 'Worst - Worst')[c(2,3)])

# invert order of axis
h_bi_all$x = -1 * h_bi_all$x

h_x_axis_label = paste0('PC1 (',round(h_pca_all$scree$data$eig[1],1),'%)')
h_y_axis_label = paste0('PC2 (',round(h_pca_all$scree$data$eig[2],1),'%)')

h_bi_all$name = gsub('Best - ', '', h_bi_all$name)
h_bi_all$name = gsub('Worst - ', '', h_bi_all$name)

h_bi_all$cols = h_bi_all$name
h_bi_all$cols[!h_bi_all$cols %in% descr_cols$descr] = ''
h_bi_all$cols = factor(h_bi_all$cols, levels=h_bi_all$cols[1:9])

# Make the actual plot
fig4h = ggplot(h_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(h_bi_all$type == 'Score', 0.8, 0), color = "black", stroke=0.1) +
  # geom_segment(aes(x = h_worst[1], y = h_worst[2], xend = h_best[1], yend = h_best[2]),
  #              linetype='solid',  alpha = 0.5, colour='#101e25', size=0.5) +
  geom_text(aes(label = name), alpha = ifelse(h_bi_all$type == 'Score', 1, 0), 
                  size = 2, segment.size = 0.25, force = 30, max.iter = 1505, 
                  max.overlaps = 200, show.legend = FALSE) +
  labs(x = h_x_axis_label, y = h_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(h_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-0.75, 0.75))+
  default_theme + theme(legend.position = 'none')



i_pca_all = data_to_biplot(df_4, val_var="Enrichment", lower_better = T)
i_bi_all = i_pca_all$bi

# invert order of axis
i_bi_all$x = -1 * i_bi_all$x

# Get xy the coordinates for the best and worst points
i_best = unlist(subset(i_bi_all, name == 'Best - Best')[c(2,3)])
i_worst = unlist(subset(i_bi_all, name == 'Worst - Worst')[c(2,3)])

i_x_axis_label = paste0('PC1 (',round(i_pca_all$scree$data$eig[1],1),'%)')
i_y_axis_label = paste0('PC2 (',round(i_pca_all$scree$data$eig[2],1),'%)')

i_bi_all$name = gsub('Best - ', '', i_bi_all$name)
i_bi_all$name = gsub('Worst - ', '', i_bi_all$name)

i_bi_all$cols = i_bi_all$name
i_bi_all$cols[!i_bi_all$cols %in% descr_cols$descr] = ''
i_bi_all$cols = factor(i_bi_all$cols, levels=i_bi_all$cols[1:9])

# Make the actual plot
fig4i = ggplot(i_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(i_bi_all$type == 'Score', 0.8, 0), color = "black", stroke=0.1) +
  # geom_segment(aes(x = i_worst[1], y = i_worst[2], xend = i_best[1], yend = i_best[2]),
  #              linetype='solid',  alpha = 0.5, colour='#101e25', size=0.5) +
  geom_text(aes(label = name), alpha = ifelse(i_bi_all$type == 'Score', 1, 0), 
                  size = 2, segment.size = 0.25, force = 30, max.iter = 1505, 
                  max.overlaps = 200, show.legend = FALSE) +
  labs(x = i_x_axis_label, y = i_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(i_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-1.5, 1.5), xlim=c(-2.5, 2.5))+
  default_theme + theme(legend.position = 'none')



j_pca_all = data_to_biplot(df_4, val_var="MCSF_hits_to_train_actives", lower_better = T)
j_bi_all = j_pca_all$bi

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
fig4j = ggplot(j_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(j_bi_all$type == 'Score', 0.8, 0), color = "black", stroke=0.1) +
  geom_text(aes(label = name), alpha = ifelse(j_bi_all$type == 'Score', 1, 0), 
                  size = 2, segment.size = 0.25, force = 50, max.iter = 1505, 
                  max.overlaps = 200, show.legend = FALSE) +
  labs(x = j_x_axis_label, y = j_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(j_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-0.15, 0.15))+
  default_theme + theme(legend.position = 'none')





k_pca_all = data_to_biplot(df_4, val_var="internal_tanimoto_hits", lower_better = F)
k_bi_all = k_pca_all$bi

# invert order of axis
k_bi_all$x = -1 * k_bi_all$x

# Get xy the coordinates for the best and worst points
k_best = unlist(subset(k_bi_all, name == 'Best - Best')[c(2,3)])
k_worst = unlist(subset(k_bi_all, name == 'Worst - Worst')[c(2,3)])

k_x_axis_label = paste0('PC1 (',round(k_pca_all$scree$data$eig[1],1),'%)')
k_y_axis_label = paste0('PC2 (',round(k_pca_all$scree$data$eig[2],1),'%)')

k_bi_all$name = gsub('Best - ', '', k_bi_all$name)
k_bi_all$name = gsub('Worst - ', '', k_bi_all$name)

k_bi_all$cols = k_bi_all$name
k_bi_all$cols[!k_bi_all$cols %in% descr_cols$descr] = ''
k_bi_all$cols = factor(k_bi_all$cols, levels=k_bi_all$cols[1:9])

# Make the actual plot
fig4k = ggplot(k_bi_all, aes(x = x, y =y)) +
  geom_point(aes(x, y, fill = cols), shape = 21,  size = 1, alpha = ifelse(k_bi_all$type == 'Score', 0.8, 0), color = "black", stroke=0.1) +
  geom_text(aes(label = name), alpha = ifelse(k_bi_all$type == 'Score', 1, 0), 
                  size = 2, segment.size = 0.25, force = 30, max.iter = 1505, 
                  max.overlaps = 30, show.legend = FALSE) +
  labs(x = k_x_axis_label, y = k_y_axis_label) +
  scale_fill_manual(values = descr_cols$cols[match(k_bi_all$cols, descr_cols$descr)]) +
  coord_cartesian(ylim=c(-0.40, 0.40))+
  default_theme + theme(legend.position = 'none')



##### plot #####

# extract a legend
lenged_fig = ggplot(df_4, aes(y = unique_scaffolds_hits_ratio, x = ranking_method, color=ranking_method))+
  geom_point() +
  scale_color_manual(values = descr_cols[1][[1]][match(levels(df_4$ranking_method), descr_cols[2][[1]])])
fig4_legend = as_ggplot(get_legend(lenged_fig))



fig4abc =  plot_grid(fig4_legend, fig4a, fig4b, fig4c,
                      labels = c('', 'a', 'b', 'c'), 
                      label_size = 10, ncol=4)


# Precision, Enrichment, Novelty, Diversity
# d, e, f, g
# h, i, j, k

# e, d, g, f
# i, h, k, j

fig4defg =  plot_grid(fig4e, fig4d, fig4g, fig4f,
                      labels = c('d', 'e', 'f', 'g'), 
                      label_size = 10, ncol=4)

fig4hijk =  plot_grid(fig4i, fig4h,fig4k, fig4j,
                      labels = c('h', 'i', 'j', 'k'), 
                      label_size = 10, ncol=4)


fig4 = plot_grid(fig4abc, fig4defg, fig4hijk,
                 rel_heights = c(1, 1.25, 1),
                 ncol=1)

fig4

# save to pdf
pdf('plots/figures/fig4.pdf', width = 180/25.4, height = 130/25.4)
print(fig4)
dev.off()

