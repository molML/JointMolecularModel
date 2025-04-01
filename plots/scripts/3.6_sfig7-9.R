# This file plots the supplementary figures sFig7-9 of the paper
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
library_inference$split = 'Library'



# Big Ridge plot
target_names = data.frame(id = c("PPARG", "Ames_mutagenicity", "ESR1_ant", "TP53", "CHEMBL1871_Ki","CHEMBL218_EC50","CHEMBL244_Ki","CHEMBL236_Ki","CHEMBL234_Ki","CHEMBL219_Ki","CHEMBL238_Ki","CHEMBL4203_Ki","CHEMBL2047_EC50","CHEMBL4616_EC50","CHEMBL2034_Ki","CHEMBL262_Ki","CHEMBL231_Ki","CHEMBL264_Ki","CHEMBL2835_Ki","CHEMBL2971_Ki","CHEMBL237_EC50","CHEMBL237_Ki","CHEMBL233_Ki","CHEMBL4792_Ki","CHEMBL239_EC50","CHEMBL3979_EC50","CHEMBL235_EC50","CHEMBL4005_Ki","CHEMBL2147_Ki","CHEMBL214_Ki","CHEMBL228_Ki","CHEMBL287_Ki","CHEMBL204_Ki","CHEMBL1862_Ki"),
                          name = c("PPARyl", "Ames", "ESR1", "TP53", "AR","CB1","FX","DOR","D3R","D4R","DAT","CLK4","FXR","GHSR","GR","GSK3","HRH1","HRH3","JAK1","JAK2","KOR (a)","KOR (i)","MOR","OX2R","PPARa","PPARym","PPARd","PIK3CA","PIM1","5-HT1A","SERT","SOR","Thrombin","ABL1"))
library_inference$dataset_name = target_names$name[match(library_inference$dataset, target_names$id)]



table(library_inference$dataset_name)
table(df_2efg$dataset_name)

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
library_inference_extended$split = factor(library_inference_extended$split, levels = c('Library', 'Test', 'OOD'))
ood_score_order = (subset(df_2efg, split == 'OOD') %>% group_by(dataset_name) %>% summarize(ood_score_mean = mean(ood_score_mean)) %>% arrange(-ood_score_mean) %>% distinct(dataset_name))$dataset_name
library_inference_extended$dataset_name = factor(library_inference_extended$dataset_name, levels=ood_score_order)


sfig7 = ggplot(library_inference_extended) +
  geom_density_ridges(aes(x = Tanimoto_to_train, y = dataset_name, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="Tanimoto to train set", y='') +
  scale_fill_manual(values = c('#79a188','#577788','#efc57b', '#4e7665', '#79a188', '#A7c6a5')) +
  scale_x_continuous(limit=c(0, 0.75)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

# save to pdf
pdf('plots/figures/sfig7.pdf', width = 90/25.4, height = 130/25.4)
print(sfig7)
dev.off()


sfig8 = ggplot(library_inference_extended) +
  geom_density_ridges(aes(x = y_unc_mean, y = dataset_name, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="H(x)", y='') +
  scale_fill_manual(values = c('#79a188','#577788','#efc57b', '#4e7665', '#79a188', '#A7c6a5')) +
  # scale_x_continuous(limit=c(-2, 8)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

# save to pdf
pdf('plots/figures/sfig8.pdf', width = 90/25.4, height = 130/25.4)
print(sfig8)
dev.off()


sfig9 = ggplot(library_inference_extended) +
  geom_density_ridges(aes(x = ood_score_mean, y = dataset_name, fill=split), alpha = 0.75, linewidth=0.35) +
  labs(x="U(x)", y='') +
  scale_fill_manual(values = c('#79a188','#577788','#efc57b', '#4e7665', '#79a188', '#A7c6a5')) +
  scale_x_continuous(limit=c(-2, 8)) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm"),)

# save to pdf
pdf('plots/figures/sfig9.pdf', width = 90/25.4, height = 130/25.4)
print(sfig9)
dev.off()






