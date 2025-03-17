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
# setwd("~/Dropbox/PycharmProjects/JointChemicalModel")

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
library_inference_summary$split = 'Library'
library_inference$split = 'Library'

library_inference_PIM1 = subset(library_inference, dataset == 'CHEMBL2147_Ki')

df_3_de <- read_csv('results/screening_mols_properties_top50.csv')
df_3_abc <- read_csv('plots/data/df_3.csv')

# Only keep the relevant ranking methods
df_3_de = subset(df_3_de, ranking_method %in% c('utopia_dist_E_min_unc',
                                                  'utopia_dist_E_min_ood',
                                                  'utopia_dist_E_min_unc_min_ood'
))

df_3_de$ranking_method = gsub('utopia_dist_E_min_unc_min_ood', 'Most reliable', df_3_de$ranking_method)
df_3_de$ranking_method = gsub('utopia_dist_E_min_unc', 'Least uncertain', df_3_de$ranking_method)
df_3_de$ranking_method = gsub('utopia_dist_E_min_ood', 'Least unfamiliar', df_3_de$ranking_method)

# Order factors
df_3_de$ranking_method = factor(df_3_de$ranking_method, levels = c(
  'Least uncertain',
  'Least unfamiliar',
  'Most reliable'
))


####  Fig3 a, b, c

dataset_name_ = 'CHEMBL262_Ki'

df_3a = subset(df_3_abc, dataset == dataset_name_ & ranking_method == 'utopia_dist_E_min_unc')
df_3a$topk_member = as.character(rank(df_3a$utopia_dist, ties.method = "min") <= 50)
fig3a = ggplot(df_3a, aes(x = y_E_, y = y_unc, fill = topk_member)) +
  geom_point(size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  scale_fill_manual(values = c('#dddddd', '#efc57b')) +
  labs(x='Expected value', y='Uncertainty')+
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.8, 0.6, 0.2, 0.2), "cm"))

df_3b = subset(df_3_abc, dataset == dataset_name_ & ranking_method == 'utopia_dist_E_min_ood')
df_3b$topk_member = as.character(rank(df_3b$utopia_dist, ties.method = "min") <= 50)
fig3b = ggplot(df_3b, aes(x = y_E_, y = ood_score, fill = topk_member)) +
  geom_point(size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  scale_fill_manual(values = c('#dddddd', '#ef9d43')) +
  labs(x='Expected value', y='Unfamiliarity')+
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.8, 0.6, 0.2, 0.2), "cm"))


####  Fig3 c - 3D plot ####

fig3c = plot_spacer()

pdf('plots/figures/fig3c.pdf', width = 4, height = 4)

df_3c = subset(df_3_abc, dataset == dataset_name_ & ranking_method == 'utopia_dist_E_min_unc_min_ood')
df_3c$topk_member = as.character(rank(df_3c$utopia_dist, ties.method = "min") <= 50)

scaled_values = 0.5 + (df_3c$utopia_dist-min(df_3c$utopia_dist))*(0.5/(max(df_3c$utopia_dist)-min(df_3c$utopia_dist)))
alpha_values = 225 * scaled_values
colors = c()
for (i in 1:length(df_3c$topk_member)){
  if (df_3c$topk_member[i] == 'TRUE'){
    colors = c(colors, rgb(col2rgb('#b75a33')[1], col2rgb('#b75a33')[2], col2rgb('#b75a33')[3], maxColorValue = 255, alpha=alpha_values[i]))
  } else {
    colors = c(colors, rgb(200, 200, 200, maxColorValue = 255, alpha=alpha_values[i]))
  }
}

scatterplot3d(df_3c$ood_score, df_3c$y_unc, df_3c$y_E_, grid = TRUE, box = TRUE, axis = T, asp = 0.8, xlim=c(-0.1, 1.6), lab=c(6, 3),  ylim=c(0.8, 0),
              pch = 20, type = "p", main = "3-way selection", angle = -145, color = colors)

dev.off()


####  Fig3 d, e - box plot ####

fig3d1 = ggplot(df_3_de, aes(y = Precision, x = ranking_method, fill=ranking_method))+
  labs(y='Precision', x='', title='') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#efc57b', '#ef9d43', '#b75a33')) +
  scale_color_manual(values = c('#efc57b', '#ef9d43', '#b75a33')) +
  scale_y_continuous(limit=c(0.25, 1)) +
  # coord_flip() +
  default_theme + theme(legend.position = 'none',
                        # axis.text.y=element_blank(),
                        axis.text.x=element_text(angle=45, hjust=1),
                        plot.margin = unit(c(0.2, 0.2, -0.6, 0), "cm"))


# wilcox.test(subset(df_3_de, ranking_method == 'Least uncertain')$Precision,
#             subset(df_3_de, ranking_method == 'Least unfamiliar')$Precision, paired=TRUE, alternative = 'two.sided')
# 
# wilcox.test(subset(df_3_de, ranking_method == 'Least uncertain')$Precision,
#             subset(df_3_de, ranking_method == 'Most reliable')$Precision, paired=TRUE, alternative = 'two.sided')
# 
# wilcox.test(subset(df_3_de, ranking_method == 'Most reliable')$Precision,
#             subset(df_3_de, ranking_method == 'Least unfamiliar')$Precision, paired=TRUE, alternative = 'two.sided')


fig3d2 = ggplot(df_3_de, aes(y = MCSF_hits_to_train_actives, x = ranking_method, fill=ranking_method))+
  labs(y='Mol. core overlap to known actives', x='', title='') +
  geom_jitter(aes(fill=ranking_method), position=position_jitterdodge(0), size=1, shape=21, alpha=0.80, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.35, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_fill_manual(values = c('#efc57b', '#ef9d43', '#b75a33')) +
  scale_color_manual(values = c('#efc57b', '#ef9d43', '#b75a33')) +
  # scale_y_continuous(limit=c(0.125, 0.5)) +
  # coord_flip() +
  default_theme + theme(legend.position = 'none',
                        # axis.text.y=element_blank(),
                        axis.text.x=element_text(angle=45, hjust=1),
                        plot.margin = unit(c(0.2, 0.2, -0.6, 0), "cm"))

# 
# wilcox.test(subset(df_3_de, ranking_method == 'Least uncertain')$MCSF_hits_to_train_actives,
#             subset(df_3_de, ranking_method == 'Least unfamiliar')$MCSF_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')
# 
# wilcox.test(subset(df_3_de, ranking_method == 'Least uncertain')$MCSF_hits_to_train_actives,
#             subset(df_3_de, ranking_method == 'Most reliable')$MCSF_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')
# 
# wilcox.test(subset(df_3_de, ranking_method == 'Most reliable')$MCSF_hits_to_train_actives,
#             subset(df_3_de, ranking_method == 'Least unfamiliar')$MCSF_hits_to_train_actives, paired=TRUE, alternative = 'two.sided')


fig3de = plot_grid(fig3d1, fig3d2, ncol=2, labels = c('d', 'e'), label_size = 10)


utopia_plots = plot_grid(fig3a, fig3b, fig3c, fig3de,
                         ncol=4, labels = c('a', 'b', 'c', ''), label_size = 10)


#### Correlations ####

# correlation between U(x) other metrics

corr_unc_ood = library_inference %>%
  group_by(dataset) %>%
  summarize(
    r_ood_unc = cor(y_unc_mean, ood_score_mean, method = 'spearman'),
    r_ood_tani = cor(Tanimoto_to_train, ood_score_mean, method = 'spearman'),
    r_ood_tani_scaff = cor(Tanimoto_scaffold_to_train, ood_score_mean, method = 'spearman'),
    r_ood_cats = cor(Cats_cos, ood_score_mean, method = 'spearman'),

    r_unc_tani = cor(Tanimoto_to_train, y_unc_mean, method = 'spearman'),
    r_unc_tani_scaff = cor(Tanimoto_scaffold_to_train, y_unc_mean, method = 'spearman'),
    r_unc_cats = cor(Cats_cos, y_unc_mean, method = 'spearman'),

    r_ood_SA = cor(SA_scores, ood_score_mean, method = 'spearman'),
    r_ood_QED = cor(QED_scores, ood_score_mean, method = 'spearman'),

    r_unc_SA = cor(SA_scores, y_unc_mean, method = 'spearman'),
    r_unc_QED = cor(QED_scores, y_unc_mean, method = 'spearman')
  ) %>% ungroup()

print(paste0('U(x) ~ H(x): r=',round(mean(corr_unc_ood$r_ood_unc),2), '±', round(se(corr_unc_ood$r_ood_unc), 2)))
print(paste0('U(x) ~ Tani: r=',round(mean(corr_unc_ood$r_ood_tani),2), '±', round(se(corr_unc_ood$r_ood_tani), 2)))
print(paste0('U(x) ~ Tani (scaff): r=',round(mean(corr_unc_ood$r_ood_tani_scaff),2), '±', round(se(corr_unc_ood$r_ood_tani_scaff), 2)))
print(paste0('U(x) ~ CATS cos: r=',round(mean(corr_unc_ood$r_ood_cats),2), '±', round(se(corr_unc_ood$r_ood_cats), 2)))

print(paste0('H(x) ~ Tani: r=',round(mean(corr_unc_ood$r_unc_tani),2), '±', round(se(corr_unc_ood$r_unc_tani), 2)))
print(paste0('H(x) ~ Tani (scaff): r=',round(mean(corr_unc_ood$r_unc_tani_scaff),2), '±', round(se(corr_unc_ood$r_unc_tani_scaff), 2)))
print(paste0('H(x) ~ CATS cos: r=',round(mean(corr_unc_ood$r_unc_cats),2), '±', round(se(corr_unc_ood$r_unc_cats), 2)))

print(paste0('U(x) ~ SA: r=',round(mean(corr_unc_ood$r_ood_SA),2), '±', round(se(corr_unc_ood$r_ood_SA), 2)))
print(paste0('U(x) ~ QED: r=',round(mean(corr_unc_ood$r_ood_QED),2), '±', round(se(corr_unc_ood$r_ood_QED), 2)))

print(paste0('H(x) ~ SA: r=',round(mean(corr_unc_ood$r_unc_SA),2), '±', round(se(corr_unc_ood$r_unc_SA), 2)))
print(paste0('H(x) ~ QED: r=',round(mean(corr_unc_ood$r_unc_QED),2), '±', round(se(corr_unc_ood$r_unc_QED), 2)))


# table(subset(library_inference, dataset == 'CHEMBL2835_Ki')$library_name)
# 452896 + 552413 + 390113 

# Get the old data from figure 2
df_2efg <- read_csv('plots/data/df_2efg.csv')
df_2efg = data.frame(
  smiles=df_2efg$smiles,
  ood_score_mean=df_2efg$ood_score,
  y_hat_mean=df_2efg$y_hat,
  y_unc_mean=df_2efg$y_unc,
  y_E_mean=df_2efg$y_E,
  mean_z_dist_mean=df_2efg$mean_z_dist,
  Tanimoto_to_train=df_2efg$Tanimoto_to_train,
  Tanimoto_scaffold_to_train=df_2efg$Tanimoto_scaffold_to_train,
  Cats_cos=df_2efg$Cats_cos,
  dataset_name=df_2efg$dataset_name,
  split=df_2efg$split)

# Add it to the inference data
library_inference_extended <- bind_rows(library_inference, df_2efg)

# Order data based on OOD score. This is not to imply any real order (datasets are independent), but just to make it less chaotic visually
library_inference_extended$split = factor(library_inference_extended$split, levels = c('Test', 'OOD', 'Library'))
ood_score_order = (subset(df_2efg, split == 'OOD') %>% group_by(dataset_name) %>% summarize(ood_score_mean = mean(ood_score_mean)) %>% arrange(-ood_score_mean) %>% distinct(dataset_name))$dataset_name
library_inference_extended$dataset_name = factor(library_inference_extended$dataset_name, levels=ood_score_order)


#### Distributions ####


#### Ridge Distributions ####


# Ridge plot
fig3f = ggplot(library_inference_extended) +
  geom_density_ridges(aes(x = Tanimoto_to_train, y=split, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="Similarity to train", y='') +
  scale_fill_manual(values = c('#577788','#efc57b', '#4E7665', '#79A188','#A7C6A5' )) +
  scale_x_continuous(limit=c(0, 0.5)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.3, 0.4, 0.2, 0), "cm"),)

# Ridge plot
fig3g = ggplot(library_inference_extended) +
  geom_density_ridges(aes(x = y_unc_mean, y=split, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="H(x)", y='') +
  scale_fill_manual(values = c('#577788','#efc57b', '#4E7665', '#79A188','#A7C6A5' )) +
  # scale_x_continuous(limit=c(0, 0.5)) +
  default_theme + theme(legend.position = 'none',
                        axis.text.y = element_blank(),
                        plot.margin = unit(c(0.3, 0.2, 0.2, 0.2), "cm"),)

# ks = ks.test(subset(library_inference_extended, split == 'Library')$y_unc_mean,
#              subset(library_inference_extended, split == 'Test')$y_unc_mean,
#              alternative="two.sided")
# print(paste0('fig2e KS test: ', ifelse(ks$p.value < 0.05, '*', 'n.s.'),' - ',  ks$p.value))


# Ridge plot
fig3h = ggplot(library_inference_extended) +
  geom_density_ridges(aes(x = ood_score_mean, y=split, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="U(x)", y='') +
  scale_fill_manual(values = c('#577788','#efc57b', '#4E7665', '#79A188','#A7C6A5' )) +
  scale_x_continuous(limit=c(-2, 8)) +
  default_theme + theme(legend.position = 'none',
                        axis.text.y = element_blank(),
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

ridge_plots = plot_grid(fig3f, fig3g, fig3h, ncol=3, labels = c('f', 'g', 'h'), label_size = 10, rel_widths = c(1,0.875,0.875))


#### 2D distributions ####


fig3k = ggplot(subset(library_inference_summary, split == 'Library'), aes(x=ood_score, y=y_unc) ) +
  labs(x='U(x)', y='H(x)') +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.01) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=0.1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=1) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon", breaks=10) +
  scale_fill_gradientn(colors = rev(c('#4E7665', '#79A188','#A7C6A5'))) +
  scale_x_continuous(limit=c(1.5, 7)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.3, 0.2, 0.2, 0.2), "cm"))


fig3l = ggplot(library_inference_PIM1, aes(x=ood_score_mean, y=y_unc_mean) ) +
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

dist_plots2 = plot_grid(plot_spacer(), fig3k, fig3l, plot_spacer(),
                        ncol=4, labels = c('', 'i', 'j', ''), label_size = 10)

fig3 = plot_grid(utopia_plots,
                 ridge_plots,
                 dist_plots2,
                 ncol=1)
fig3

# save to pdf
pdf('plots/figures/fig3_.pdf', width = 180/25.4, height = 135/25.4)
print(fig3)
dev.off()


#### Look for example molecules ####
# We calculated the utopia distance and then cherry picked from the top molecules for molecules that are obvious
# enough in their structure to make the point without having to explain too much chemistry background in the paper

# calc_utopia_dist <- function(y_E, confidence, param3 = NULL, maximize_param1 = TRUE, maximize_param2 = TRUE, maximize_param3 = TRUE) {
#   # Convert inputs to numeric vectors (if not already)
#   y_E <- as.numeric(y_E)
#   confidence <- as.numeric(confidence)
#   
#   # Calculate max and min for normalization
#   E_max <- max(y_E)
#   E_min <- min(y_E)
#   conf_max <- max(confidence)
#   conf_min <- min(confidence)
#   
#   
#   # Normalize bioactivity based on toggle
#   if (maximize_param1) {
#     norm_bio <- (E_max - y_E) / (E_max - E_min)  # Higher is better
#   } else {
#     norm_bio <- (y_E - E_min) / (E_max - E_min)  # Lower is better
#   }
#   
#   # Normalize confidence based on toggle
#   if (maximize_param2) {
#     norm_conf <- (conf_max - confidence) / (conf_max - conf_min)  # Higher is better
#   } else {
#     norm_conf <- (confidence - conf_min) / (conf_max - conf_min)  # Lower is better
#   }
#   
#   # If param3 is provided, include it in the distance calculation
#   if (!is.null(param3)) {
#     param3 <- as.numeric(param3)
#     param3_max <- max(param3)
#     param3_min <- min(param3)
#     
#     # Normalize param3 based on toggle
#     if (maximize_param3) {
#       norm_param3 <- (param3_max - param3) / (param3_max - param3_min)  # Higher is better
#     } else {
#       norm_param3 <- (param3 - param3_min) / (param3_max - param3_min)  # Lower is better
#     }
#     
#     # Compute the Euclidean distance in 3D
#     dist_ranking <- sqrt(norm_bio^2 + norm_conf^2 + norm_param3^2)
#   } else {
#     # Compute the Euclidean distance in 2D
#     dist_ranking <- sqrt(norm_bio^2 + norm_conf^2)
#   }
#   
#   return(dist_ranking)
# }
# 
# 
# library_inference_summary_hits = subset(library_inference_summary, y_E >= 0.5)
# 
# library_inference_summary_hits$max_U_max_H = calc_utopia_dist(library_inference_summary_hits$ood_score_mean, library_inference_summary_hits$y_unc_mean, 
#                                                               maximize_param1 = TRUE, maximize_param2 = TRUE)
# 
# library_inference_summary_hits$max_U_min_H = calc_utopia_dist(library_inference_summary_hits$ood_score_mean, library_inference_summary_hits$y_unc_mean, 
#                                                               maximize_param1 = TRUE, maximize_param2 = F)
# 
# library_inference_summary_hits$min_U_min_H = calc_utopia_dist(library_inference_summary_hits$ood_score_mean, library_inference_summary_hits$y_unc_mean, 
#                                                               maximize_param1 = F, maximize_param2 = F)
# 
# library_inference_summary_hits$min_U_max_H = calc_utopia_dist(library_inference_summary_hits$ood_score_mean, library_inference_summary_hits$y_unc_mean, 
#                                                               maximize_param1 = F, maximize_param2 = TRUE)
#
#
# library_inference_summary_hits_PIM1 = subset(library_inference_PIM1, y_E_mean >= 0.5)
# 
# library_inference_summary_hits_PIM1$max_U_max_H = calc_utopia_dist(library_inference_summary_hits_PIM1$ood_score_mean, library_inference_summary_hits_PIM1$y_unc_mean, 
#                                                               maximize_param1 = TRUE, maximize_param2 = TRUE)
# 
# library_inference_summary_hits_PIM1$max_U_min_H = calc_utopia_dist(library_inference_summary_hits_PIM1$ood_score_mean, library_inference_summary_hits_PIM1$y_unc_mean, 
#                                                               maximize_param1 = TRUE, maximize_param2 = F)
# 
# library_inference_summary_hits_PIM1$min_U_min_H = calc_utopia_dist(library_inference_summary_hits_PIM1$ood_score_mean, library_inference_summary_hits_PIM1$y_unc_mean, 
#                                                               maximize_param1 = F, maximize_param2 = F)
# 
# library_inference_summary_hits_PIM1$min_U_max_H = calc_utopia_dist(library_inference_summary_hits_PIM1$ood_score_mean, library_inference_summary_hits_PIM1$y_unc_mean, 
#                                                               maximize_param1 = F, maximize_param2 = TRUE)
