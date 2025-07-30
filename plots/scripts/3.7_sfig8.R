# This file plots the supplementary figure sFig8 of the paper
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

# the color scheme I used throughout the paper
cols = c('#577788','#97a4ab','#ef9d43','#efc57b', '#578d88', '#99beae')

# AUC calculation function (trapezoidal rule)
calc_auc <- function(row_values, wavelengths) {
  sum(diff(wavelengths) * (head(row_values, -1) + tail(row_values, -1)) / 2)
}

# subtract data with the blank values, compute AUC, normalize by negative control
process_raw_data <- function(df_curves){
  
  df_curves = subset(df_curves, Keep != 'Discard')
  
  # Identify luminescence columns
  lum_cols2 <- grep("Luminescence", colnames(df_curves), value = TRUE)
  
  # Extract numeric wavelengths from column names
  wavelengths2 <- as.numeric(gsub("Luminescence \\((\\d+) nm\\)", "\\1", lum_cols2))
  
  # Find blank values
  curves_blank_vals <- df_curves %>%
    filter(Type == "Blank") %>%
    select(Protein, starts_with("Luminescence")) %>%
    group_by(Protein) %>%
    summarise(across(all_of(lum_cols2), mean), .groups = "drop")
  
  
  # Step 2: subtract blank values from data rows
  df_curves_norm <- df_curves %>%
    filter(Type != "Blank") %>%
    rowwise() %>%
    mutate(across(
      starts_with("Luminescence"),
      ~ . - curves_blank_vals[curves_blank_vals$Protein == Protein, cur_column()][[1]]
    )) %>%
    ungroup()
  
  # Add AUC column
  df_curves_norm$AUC <- apply(df_curves_norm[, lum_cols2], 1, calc_auc, wavelengths = wavelengths2)
  
  # Find negative control values (100% protein activity)
  neg_control <- df_curves_norm %>%
    filter(Type == "Negative control") %>%
    group_by(Protein, Type) %>%
    summarise(AUC = mean(AUC))
  
  # convert luminescence tot activity by normalizing for 100% activity (just protein wells)
  df_curves_norm <- df_curves_norm %>%
    group_by(Protein) %>%
    mutate(Activity = AUC * 100 / subset(neg_control, Protein == Protein)$AUC) %>% 
    ungroup()
  
  df_curves_norm$Replicate = as.character(df_curves_norm$Replicate)
  
  # Convert dose to log molar dose
  df_curves_norm$Dose_molar = as.numeric(df_curves_norm$`Concentration (µM)`) *  1e-6
  
  # Remove the data of just the protein, as we won't use this for curve fitting.
  df_curves_norm = subset(df_curves_norm, Compound != '-')
  
  return(df_curves_norm)
}


setwd("~/Dropbox/PycharmProjects/JointMolecularModel")


#### Point screening ####

# Load the data
pim1 <- read_csv('plots/data/hit screening/25_06_25_pim1_long.csv')
cdk1 <- read_csv('plots/data/hit screening/24_06_25_cdk1_long.csv')

# inhibition per group
screening_lookup_table <- read_csv('plots/data/screening_lookup_table.csv')
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
  filter(Type == "Blank") %>%
  select(Protein, starts_with("Luminescence"))

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
  left_join(neg_control_point_screen %>% select(Protein, AUC_NegControl = AUC),
            by = "Protein") %>%
  mutate(Activity = AUC * 100 / AUC_NegControl) %>% 
  select(Type, Compound, Protein, Activity)

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

# convert to long format
df_long <- df_norm %>%
  pivot_longer(
    cols = all_of(lum_cols1),
    names_to = "Wavelength",
    values_to = "Luminescence"
  ) %>%
  mutate(
    Wavelength = as.numeric(gsub("Luminescence \\((\\d+) nm\\)", "\\1", Wavelength))
  )
df_long = subset(df_long, AUC > 0)


# Get the compounds with the best AUC (inc the reference compound)
n_top = 7
top_bottom_compounds <- df_norm %>%
  group_by(Protein) %>%
  filter(Type %in% c("Screen", "Positive control")) %>%
  arrange(AUC) %>%
  slice(c(1:n_top)) %>%
  select(c(Compound, Protein))

# Find wavelength with maximum luminescence for each compound (we'll put the label there)
df_labels_belly <- df_long %>%
  group_by(Protein, Compound) %>%
  filter(Type %in% c("Screen", "Positive control")) %>%
  slice_max(order_by = Luminescence, n = 1, with_ties = FALSE) %>%
  ungroup()

# Make the dataframes used for the plots
df_pim1 = subset(df_long, Protein == 'PIM1')
pim1_labels = subset(df_labels_belly, Protein == 'PIM1' & Compound %in% subset(top_bottom_compounds, Protein == 'PIM1')$Compound)

df_cdk1 = subset(df_long, Protein == 'CDK1')
cdk1_labels = subset(df_labels_belly, Protein == 'CDK1' & Compound %in% subset(top_bottom_compounds, Protein == 'CDK1')$Compound)


sfig8a = ggplot(df_pim1, aes(x = Wavelength, y = Luminescence, group = interaction(Type, Compound))) +
  # All regular samples: colored by AUC
  geom_line(data = df_pim1 %>% filter(!Type %in% c("Positive control", "Negative control")),
            aes(color = AUC), linewidth = 0.35) +
  # Highlight negative control
  geom_line(data = df_pim1 %>% filter(Type == "Negative control"),
            color = "#16263a", linewidth = 0.5, linetype = "solid") +
  # Highlight positive control
  geom_line(data = df_pim1 %>% filter(Type == "Positive control"),
            color = "#16263a", linewidth = 0.5, linetype = "dashed") +
  labs(y = "Luminescence", x = "Wavelength (nm)") +
  # Center color scale at negative control AUC
  scale_color_gradient2(midpoint = subset(df_pim1, Type == "Negative control")$AUC[1],
                        low = "#4E7665", mid = "white", high = "#bbbbbb",name = "AUC") +
  geom_text(data = pim1_labels, aes(label = Compound), size = 2.5, show.legend = FALSE) +
  # coord_cartesian(xlim=c(488, 653))+
  default_theme + theme(legend.position = 'none')


sfig8b = ggplot(df_cdk1, aes(x = Wavelength, y = Luminescence, group = interaction(Type, Compound))) +
  # All regular samples: colored by AUC
  geom_line(data = df_cdk1 %>% filter(!Type %in% c("Positive control", "Negative control")),
            aes(color = AUC), linewidth = 0.35) +
  # Highlight negative control
  geom_line(data = df_cdk1 %>% filter(Type == "Negative control"),
            color = "#16263a", linewidth = 0.5, linetype = "solid") +
  # Highlight positive control
  geom_line(data = df_cdk1 %>% filter(Type == "Positive control"),
            color = "#16263a", linewidth = 0.5, linetype = "dashed") +
  labs(y = "Luminescence", x = "Wavelength (nm)") +
  # Center color scale at negative control AUC
  scale_color_gradient2(midpoint = subset(df_cdk1, Type == "Negative control")$AUC[1],
                        low = "#4E7665", mid = "white", high = "#bbbbbb",name = "AUC") +
  geom_text(data = cdk1_labels, aes(label = Compound), size = 2.5, show.legend = FALSE) +
  # coord_cartesian(xlim=c(488, 653))+
  default_theme + theme(legend.position = 'none')

legend_plot = ggplot(df_heatmap, aes(x = Compound, y = Protein, fill = Activity)) +
  geom_tile() +
  scale_fill_gradient2(midpoint=100, low = "#4E7665", mid = "white", high = "#bbbbbb",
                       breaks = c(0, 25, 50, 75, 100), name = "Activity")



sfig8 = plot_grid(sfig8a, sfig8b, rel_widths = c(1, 1, 0.75),
                   ncol=3, align = "h", axis = "ltb", 
                   labels = c('a', 'b', ''),
                   label_size=10)


pdf('plots/figures/sfig8.pdf', width = 180/25.4, height = 40/25.4)
print(sfig11)
dev.off()