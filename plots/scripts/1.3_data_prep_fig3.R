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

se <- function(x, na.rm = FALSE) {sd(x, na.rm=na.rm) / sqrt(sum(1*(!is.na(x))))}

# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointMolecularModel")

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
# "U(x) ~ H(x): r=-0.03±0.03"
print(paste0('U(x) ~ Tani: r=',round(mean(corr_unc_ood$r_ood_tani),2), '±', round(se(corr_unc_ood$r_ood_tani), 2)))
# "U(x) ~ Tani: r=-0.39±0.01"
print(paste0('U(x) ~ Tani (scaff): r=',round(mean(corr_unc_ood$r_ood_tani_scaff),2), '±', round(se(corr_unc_ood$r_ood_tani_scaff), 2)))
# "U(x) ~ Tani (scaff): r=-0.14±0.02"
print(paste0('U(x) ~ CATS cos: r=',round(mean(corr_unc_ood$r_ood_cats),2), '±', round(se(corr_unc_ood$r_ood_cats), 2)))
# "U(x) ~ CATS cos: r=-0.12±0.02"

print(paste0('H(x) ~ Tani: r=',round(mean(corr_unc_ood$r_unc_tani),2), '±', round(se(corr_unc_ood$r_unc_tani), 2)))
# "H(x) ~ Tani: r=-0.01±0.01"
print(paste0('H(x) ~ Tani (scaff): r=',round(mean(corr_unc_ood$r_unc_tani_scaff),2), '±', round(se(corr_unc_ood$r_unc_tani_scaff), 2)))
# "H(x) ~ Tani (scaff): r=0±0.02"
print(paste0('H(x) ~ CATS cos: r=',round(mean(corr_unc_ood$r_unc_cats),2), '±', round(se(corr_unc_ood$r_unc_cats), 2)))
# "H(x) ~ CATS cos: r=0±0.02"

print(paste0('U(x) ~ SA: r=',round(mean(corr_unc_ood$r_ood_SA),2), '±', round(se(corr_unc_ood$r_ood_SA), 2)))
# "U(x) ~ SA: r=0.01±0.02"
print(paste0('U(x) ~ QED: r=',round(mean(corr_unc_ood$r_ood_QED),2), '±', round(se(corr_unc_ood$r_ood_QED), 2)))
# "U(x) ~ QED: r=-0.04±0.03"

print(paste0('H(x) ~ SA: r=',round(mean(corr_unc_ood$r_unc_SA),2), '±', round(se(corr_unc_ood$r_unc_SA), 2)))
# "H(x) ~ SA: r=-0.01±0.03"
print(paste0('H(x) ~ QED: r=',round(mean(corr_unc_ood$r_unc_QED),2), '±', round(se(corr_unc_ood$r_unc_QED), 2)))
# "H(x) ~ QED: r=0.01±0.04"

# table(subset(library_inference, dataset == 'CHEMBL2835_Ki')$library_name)
# 452896 + 552413 + 390113 

# Get the data from figure 2
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


# Remove the SMILES column. This is done because the commercial screening libraries
# do not give explicit permissions for re-distributing their molecular structures.
# If you download their original data yourself, you can of course back track everything,
# but for the sake of sharing data for this paper, I removed the SMILES for the source
# files of Fig. 3.  

# I only keep the columns required for plotting to keep file size manageable

library_inference_extended <- library_inference_extended %>% 
  select(Tanimoto_to_train, y_unc_mean, ood_score_mean, split)
write.csv(library_inference_extended, 'plots/data/df_3_abc.csv', row.names = FALSE)

library_inference_summary <- subset(library_inference_summary, split == 'Library') %>% 
  select(ood_score, y_unc)
write.csv(library_inference_summary, 'plots/data/df_3_d.csv', row.names = FALSE)

library_inference_PIM1 <- library_inference_PIM1 %>% 
  select(ood_score_mean, y_unc_mean)
write.csv(library_inference_PIM1, 'plots/data/df_3_e.csv', row.names = FALSE)
