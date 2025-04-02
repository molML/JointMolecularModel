# This file processes all library inference data.
#
# Derek van Tilborg
# Eindhoven University of Technology
# March 2024

#### libraries ####

library(readr)
library(ggplot2)
library(dplyr)
library(cowplot)
library(ggridges)
library(viridis)
library(hrbrthemes)
library(data.table)
library(patchwork)
library(caret)
library(stringr)
library(ggrepel)
library(factoextra)
library(tidyr)


se <- function(x, na.rm = FALSE) {sd(x, na.rm=na.rm) / sqrt(sum(1*(!is.na(x))))}

summarize_library <- function(df){
  df %>%
    group_by(smiles) %>%
    summarize(
      ood_score_mean = mean(reconstruction_loss),
      ood_score_se = se(reconstruction_loss),
      y_hat_mean = mean(y_hat),
      y_hat_se = se(y_hat),
      y_unc_mean = mean(y_unc),
      y_unc_se = se(y_unc),
      y_E_mean = mean(y_E),
      y_E_se = se(y_E),
      mean_z_dist_mean = mean(mean_z_dist),
      mean_z_dist_se = se(mean_z_dist)
    )
}

setwd("~/Dropbox/PycharmProjects/JointMolecularModel")

dataset_names = gsub('_asinex_inference.csv', '', list.files('results/screening_libraries', pattern='asinex'))

# Loop over all inference data for every model. Average out over the 10 different seeds
library_inference <- list()
for (dataset_i in 1:(length(dataset_names))){
  dataset = dataset_names[dataset_i]
  print(paste(dataset_i, dataset))

  asinex = read_csv(paste0("results/screening_libraries/", list.files('results/screening_libraries', pattern=paste0(dataset, '_asinex'))))
  asinex = summarize_library(asinex)
  asinex$library_name = 'asinex'

  specs = read_csv(paste0("results/screening_libraries/", list.files('results/screening_libraries', pattern=paste0(dataset, '_specs'))))
  specs = summarize_library(specs)
  specs$library_name = 'specs'

  enamine = read_csv(paste0("results/screening_libraries/", list.files('results/screening_libraries', pattern=paste0(dataset, '_enamine_hit_locator'))))
  enamine = summarize_library(enamine)
  enamine$library_name = 'enamine'

  together = rbind(asinex, specs, enamine)
  together$dataset = dataset

  rm(asinex, specs, enamine)
  
  library_inference[[length(library_inference)+1]] <- together
  
}
library_inference <- do.call(rbind, library_inference)


all_distances <- list()
for (dataset_i in 1:(length(dataset_names))){
  
  dataset = dataset_names[dataset_i]
  # Load distance data
  distance_file_path = list.files('data/datasets_with_metrics', pattern=paste0(dataset, '_library'))
  distances = read_csv(paste0("data/datasets_with_metrics/", distance_file_path))
  distances$dataset = dataset
  
  all_distances[[length(all_distances)+1]] <- distances
  
  rm(distances)
}
all_distances <- do.call(rbind, all_distances)

# match distances with inference data
library_inference <- library_inference %>%
  left_join(all_distances, by = c("smiles", 'dataset'))

# Save 
write.csv(library_inference, 'results/screening_libraries/all_inference_data.csv', row.names = FALSE)

