# This file prepares the data for fig4 of the paper
#
# Derek van Tilborg
# Eindhoven University of Technology
# Juli 2025
# loading some libraries

library(readr)
library(drc)
library(dplyr)
library(tidyr)

# AUC calculation function (trapezoidal rule)
calc_auc <- function(row_values, wavelengths) {
  sum(diff(wavelengths) * (head(row_values, -1) + tail(row_values, -1)) / 2)
}

setwd("~/Dropbox/PycharmProjects/JointMolecularModel")


#### Point screening ####

# Load the data
pim1 <- read_csv('results/prospective/hit screening/25_06_25_pim1_long.csv')
cdk1 <- read_csv('results/prospective/hit screening/24_06_25_cdk1_long.csv')

# inhibition per group
screening_lookup_table <- read_csv('results/prospective/screening_lookup_table.csv')
screening_lookup_table$Protein = factor(screening_lookup_table$Protein, level=c('PIM1', 'CDK1'))

df = rbind(pim1, cdk1)

# rename the compounds from their intermediate ID (the one I used in the lab) to the ID Im using in the paper and PhD thesis
df$Compound = screening_lookup_table$Cpd_ID[match(df$Compound, screening_lookup_table$Intermediate_cpd_ID)]

# Identify luminescence columns
lum_cols1 <- grep("Luminescence", colnames(df), value = TRUE)

# Extract numeric wavelengths from column names
wavelengths1 <- as.numeric(gsub("Luminescence \\((\\d+) nm\\)", "\\1", lum_cols1))

# Average replicates
df_avg <- df %>%
  group_by(Type, Compound, Protein) %>%
  summarise(across(all_of(lum_cols1), mean), .groups = "drop")

# Find blank values
blank_vals <- df_avg %>%
  dplyr::filter(Type == "Blank") %>%
  dplyr::select(Protein, starts_with("Luminescence"))

# Step 2: subtract blank values from data rows
df_norm <- df_avg %>%
  filter(Type != "Blank") %>%
  rowwise() %>%
  mutate(across(
    starts_with("Luminescence"),
    ~ . - blank_vals[blank_vals$Protein == Protein, cur_column()][[1]]
  )) %>%
  ungroup()

# Add AUC column
df_norm$AUC <- apply(df_norm[, lum_cols1], 1, calc_auc, wavelengths = wavelengths1)

# Find negative control values (100% protein activity)
neg_control_point_screen <- df_norm %>%
  filter(Type == "Negative control") %>%
  group_by(Protein, Type) %>%
  summarise(AUC = mean(AUC))%>% 
  ungroup()

# convert luminescence tot activity by normalizing for 100% activity (just protein wells)
df_heatmap <- df_norm %>%
  left_join(neg_control_point_screen %>% dplyr::select(Protein, AUC_NegControl = AUC),
            by = "Protein") %>%
  mutate(Activity = AUC * 100 / AUC_NegControl) %>% 
  dplyr::select(Type, Compound, Protein, Activity)

# Get the compounds with the best AUC (inc the reference compound)
n_top = 7
top_bottom_compounds <- df_norm %>%
  group_by(Protein) %>%
  filter(Type %in% c("Screen", "Positive control")) %>%
  arrange(AUC) %>%
  slice(c(1:n_top)) %>%
  dplyr::select(c(Compound, Protein))


# Build the dataframe used for plots b, c, e, f, g
df_heatmap = subset(df_heatmap, Compound != '-')

# Add the selection method both in full name and A, B, C (per the paper)
df_heatmap$method = screening_lookup_table$Method[match(df_heatmap$Compound, screening_lookup_table$Cpd_ID)]
df_heatmap$method_ABC = NA
df_heatmap$method_ABC[which(df_heatmap$method == 'Most_uncertain_least_unfamiliar')] = 'A'
df_heatmap$method_ABC[which(df_heatmap$method == 'Least_uncertain_least_unfamiliar')] = 'B'
df_heatmap$method_ABC[which(df_heatmap$method == 'Least_uncertain_most_unfamiliar')] = 'C'
df_heatmap$method_ABC = factor(df_heatmap$method_ABC, levels=c('A', 'B', 'C'))

# Add the other stats
df_heatmap$Tanimoto_to_dataset_max = screening_lookup_table$Tanimoto_to_dataset_max[match(df_heatmap$Compound, screening_lookup_table$Cpd_ID)]

# add the proper labels (i.e., turn all compounds that are not in the top list into NA)
df_heatmap$label_scatter = df_heatmap$Compound
df_heatmap$label_scatter[which(!df_heatmap$label_scatter %in% top_bottom_compounds$Compound)] = NA

# save source files
write.csv(screening_lookup_table, 'plots/data/df_4_ad.csv', row.names = FALSE)
write.csv(df_heatmap, 'plots/data/df_4_bcefg.csv', row.names = FALSE)
