# This file plots the main results (fig4) of the paper
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

# Make the dataframes used for all the plots
df_pim1 = subset(df_long, Protein == 'PIM1')
pim1_labels = subset(df_labels_belly, Protein == 'PIM1' & Compound %in% subset(top_bottom_compounds, Protein == 'PIM1')$Compound)

df_cdk1 = subset(df_long, Protein == 'CDK1')
cdk1_labels = subset(df_labels_belly, Protein == 'CDK1' & Compound %in% subset(top_bottom_compounds, Protein == 'CDK1')$Compound)

df_heatmap_pim = subset(df_heatmap, Protein == 'PIM1')
pim_cpd_order = c("AZD1208", as.character(1:30))
df_heatmap_pim$Compound = factor(df_heatmap_pim$Compound, level=pim_cpd_order)
df_heatmap_pim$label_scatter = pim1_labels$Compound[match(df_heatmap_pim$Compound, pim1_labels$Compound)]

df_heatmap_cdk = subset(df_heatmap, Protein == 'CDK1')
cdk_cpd_order = c("Dinaciclib", as.character(31:60))
df_heatmap_cdk$Compound = factor(df_heatmap_cdk$Compound, level=cdk_cpd_order)
df_heatmap_cdk$label_scatter = cdk1_labels$Compound[match(df_heatmap_cdk$Compound, cdk1_labels$Compound)]



#### fig 4 ####

col_a = '#b75a33'
col_b = '#577788'
col_c = '#efc57b'

fig4a = ggplot(subset(screening_lookup_table, Protein %in% c('PIM1')),
               aes(x=unfamiliarity, y=y_unc, fill=Method))+
  geom_point(size=1.5, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  scale_fill_manual(values=c(col_b, col_c, col_a))+
  labs(x='U(x)', y='H(x)')+
  default_theme + theme(legend.position = 'none')


# PIM1 activity plot

fig4b = ggplot(subset(df_heatmap_pim, Type == 'Screen'), aes(x = Tanimoto_to_dataset_max, y = Activity, fill=method_ABC))+
  geom_point(size=1.5, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  geom_text(aes(label = label_scatter), size=2, nudge_x = 0.01, nudge_y = 0.01, color = "black") +
  scale_y_continuous(limit=c(0, 120), breaks=c(0,25,50,75,100, 120)) +
  scale_x_continuous(limits=c(0.15, 0.45)) +
  labs(y='PIM1 activity (%)', x='Max similarity to training data', title='') +
  scale_fill_manual(values=c(col_a, col_b, col_c))+
  default_theme + theme(legend.position = 'none')

fig4c = ggplot(subset(df_heatmap_pim, Type == 'Screen'), aes(x = method_ABC, y = Activity, fill=method_ABC))+
  geom_jitter(aes(fill=method_ABC), position=position_jitterdodge(0), size=1, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.5, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_y_continuous(limit=c(0, 120), breaks=c(0,25,50,75,100, 120)) +
  labs(x='Method', title='', y='PIM1 activity (%)') +
  geom_hline(yintercept=100, linewidth = 0.5, linetype = "solid") +
  geom_hline(yintercept=min(df_heatmap_pim$Activity), linewidth = 0.5, linetype = "dashed") +
  scale_fill_manual(values=c(col_a, col_b, col_c))+
  default_theme + theme(legend.position = 'none',
                        axis.title.y=element_blank(),
                        plot.margin = margin(t = 0,  # Top margin
                                             # r = 0,  # Right margin
                                             b = 5,  # Bottom margin
                                             l = 0)) # trbl



wilcox.test(subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'A')$Activity,
            subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'B')$Activity, 
            paired=F, alternative = 'two.sided')

wilcox.test(subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'B')$Activity,
            subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'C')$Activity, 
            paired=F, alternative = 'two.sided')

wilcox.test(subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'A')$Activity,
            subset(df_heatmap_pim, Type == 'Screen' & method_ABC == 'C')$Activity, 
            paired=F, alternative = 'two.sided')


fig4d = ggplot(subset(screening_lookup_table, Protein %in% c('CDK1')),
               aes(x=unfamiliarity, y=y_unc, fill=Method))+
  geom_point(size=1.5, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  scale_fill_manual(values=c(col_b, col_c, col_a))+
  labs(x='U(x)', y='H(x)')+
  default_theme + theme(legend.position = 'none')



# CDK1 activity plot
fig4e = ggplot(subset(df_heatmap_cdk, Type == 'Screen'), aes(x = Tanimoto_to_dataset_max, y = Activity, fill=method_ABC))+
  geom_point(size=1.5, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  geom_text(aes(label = label_scatter), size=2, nudge_x = 0.01, nudge_y = 0.01, color = "black") +
  scale_y_continuous(limit=c(0, 120), breaks=c(0,25,50,75,100, 120)) +
  scale_x_continuous(limits=c(0.15, 0.45)) +
  labs(y='CDK1 activity (%)', x='Max similarity to training data', title='') +
  scale_fill_manual(values=c(col_a, col_b, col_c))+
  default_theme + theme(legend.position = 'none',
                        # axis.title.y=element_blank(),
                        # axis.text.y=element_blank(),
                        # axis.ticks.y=element_blank(),
  )

fig4f = ggplot(subset(df_heatmap_cdk, Type == 'Screen'), aes(x = method_ABC, y = Activity, fill=method_ABC))+
  geom_jitter(aes(fill=method_ABC), position=position_jitterdodge(0), size=1, shape=21, alpha=0.8, color = "black", stroke = 0.1) +
  geom_boxplot(alpha=0.5, outlier.size = 0, position = position_dodge(0.75), width = 0.5, outlier.shape=NA,
               varwidth = FALSE, lwd=0.3, fatten=0.75) +
  scale_y_continuous(limit=c(0, 120), breaks=c(0,25,50,75,100, 120)) +
  labs(x='Method', title='', y='CDK1 activity (%)') +
  geom_hline(yintercept=100, linewidth = 0.5, linetype = "solid") +
  geom_hline(yintercept=min(df_heatmap_pim$Activity), linewidth = 0.5, linetype = "dashed") +
  scale_fill_manual(values=c(col_a, col_b, col_c))+
  default_theme + theme(legend.position = 'none',
                        axis.title.y=element_blank(),
                        # axis.text.y=element_blank(),
                        # axis.ticks.y=element_blank(),
                        plot.margin = margin(t = 0,  # Top margin
                                             # r = 0,  # Right margin
                                             b = 5,  # Bottom margin
                                             l = 0)
  ) # trbl

wilcox.test(subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'A')$Tanimoto_to_dataset_max,
            subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'B')$Tanimoto_to_dataset_max, 
            paired=F, alternative = 'two.sided')

wilcox.test(subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'B')$Tanimoto_to_dataset_max,
            subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'C')$Tanimoto_to_dataset_max, 
            paired=F, alternative = 'two.sided')

wilcox.test(subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'A')$Tanimoto_to_dataset_max,
            subset(df_heatmap_cdk, Type == 'Screen' & method_ABC == 'C')$Tanimoto_to_dataset_max, 
            paired=F, alternative = 'two.sided')


df_heatmap_pim$label = paste0(round(df_heatmap_pim$Activity, 0), '%')
fig4g1 = ggplot(df_heatmap_pim, aes(x = Compound, y = Protein, fill = Activity)) +
  geom_tile() +
  scale_fill_gradient2(midpoint=100, low = "#4E7665", mid = "white", high = "#bbbbbb", name = "Activity")+
  geom_text(aes(label = label), color = "#101e25", size = 2) +
  default_theme + theme(legend.position = 'none',
                        axis.title.x=element_blank(),
                        axis.title.y=element_blank(),
                        axis.ticks.y=element_blank())

df_heatmap_cdk$label = paste0(round(df_heatmap_cdk$Activity, 0), '%')
fig4g2 = ggplot(df_heatmap_cdk, aes(x = Compound, y = Protein, fill = Activity)) +
  geom_tile() +
  scale_fill_gradient2(midpoint=100, low = "#4E7665", mid = "white", high = "#bbbbbb", name = "Activity")+
  geom_text(aes(label = label), color = "#101e25", size = 2) +
  default_theme + theme(legend.position = 'none',
                        axis.title.x=element_blank(),
                        axis.title.y=element_blank(),
                        axis.ticks.y=element_blank())

legend_plot = ggplot(df_heatmap, aes(x = Compound, y = Protein, fill = Activity)) +
  geom_tile() +
  scale_fill_gradient2(midpoint=100, low = "#4E7665", mid = "white", high = "#bbbbbb",
                       breaks = c(0, 25, 50, 75, 100), name = "Activity")

legend <- cowplot::get_legend(legend_plot)
ggdraw(legend)


fig_4g = plot_grid(fig4g1, fig4g2, ncol=1, align = "hv", axis = "tblr",
                   rel_heights = c(1, 1), labels = c('e', ''),
                   label_size=10)

fig_4g = plot_grid(fig_4g, legend, ncol=1, rel_heights = c(1, 0.075))

fig_4abcdef = plot_grid(fig4a, fig4b, fig4c, fig4d, fig4e, fig4f, ncol=6, align = "h", axis = "ltb",
                        rel_widths = c(0.9, 0.9, 0.3, 0.9, 0.9, 0.3), labels = c('a', 'b', 'c', 'd', 'e', 'f'),
                        label_size=10)

fig_4 = plot_grid(fig_4abcdef, fig_4g, rel_heights = c(1, 0.72), ncol=1)
fig_4


pdf('plots/figures/fig4.pdf', width = 180/25.4, height = 70/25.4)
print(fig_4)
dev.off()







