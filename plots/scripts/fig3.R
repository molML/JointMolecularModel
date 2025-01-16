# This file plots the main results (Fig3) of the paper
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


#### Utils ####

se <- function(x, na.rm = FALSE) {sd(x, na.rm=na.rm) / sqrt(sum(1*(!is.na(x))))}

compute_balanced_accuracy <- function(y_true, y_hat) {
  # Ensure that y_true and y_hat are factors with the same levels
  y_true <- factor(y_true)
  y_hat <- factor(y_hat, levels = levels(y_true))
  
  # Create confusion matrix
  cm <- table(y_true, y_hat)
  
  # Calculate the true positive rate (sensitivity) for each class
  sensitivity_per_class <- diag(cm) / rowSums(cm)
  
  # Calculate the balanced accuracy
  balanced_accuracy <- mean(sensitivity_per_class, na.rm = TRUE)
  
  return(balanced_accuracy)
}

compute_precision <- function(y_true, y_hat) {
  # Ensure that y_true and y_hat are factors with the same levels
  y_true <- factor(y_true)
  y_hat <- factor(y_hat, levels = levels(y_true))
  
  confusion = data.frame(table(y_true,y_hat))
  
  FP = confusion[2, ]$Freq
  TP = confusion[4, ]$Freq
  
  PPV = TP / (TP + FP)
  
  return(PPV)
  
}

compute_tpr <- function(y_true, y_hat) {
  # Ensure that y_true and y_hat are factors with the same levels
  y_true <- factor(y_true)
  y_hat <- factor(y_hat, levels = levels(y_true))
  
  confusion = data.frame(table(y_true,y_hat))
  
  TP = confusion[4, ]$Freq
  FN = confusion[3, ]$Freq
  
  TPR  = TP / (TP + FN)
  
  return(TPR)
}


#### load data ####

# Load the data and change some names/factors
setwd("~/Dropbox/PycharmProjects/JointChemicalModel")

df_3abc <- read_csv('plots/data/df_3abc.csv')

df_3abc$reliability_method = factor(df_3abc$reliability_method, levels = c("Scaffold sim", "Mol core overlap", "Pharmacophore sim", "Embedding dist", "Uncertainty","Unfamiliarity"))
df_3abc$bin = factor(df_3abc$bin)

#### ranking correlation ####

calibration_summary <- df_3abc %>% 
  group_by(dataset_name, reliability_method, bin) %>%
  summarize(
    balanced_acc = compute_balanced_accuracy(y, 1*(y_hat>0.5)),
    tpr = compute_tpr(y, 1*(y_hat>0.5)),
    precision = compute_precision(y, 1*(y_hat>0.5)),
    MCSF = mean(MCSF_),
    Cats_cos = mean(Cats_cos_),
    Tanimoto_scaffold_to_train = mean(Tanimoto_scaffold_to_train_),
    n_bin = n(),
    reliability_bin = mean(reliability)
  ) %>% ungroup() %>% drop_na() %>% 
  group_by(reliability_method, bin) %>%
  summarise(
    balanced_acc_mean = mean(balanced_acc, na.rm = TRUE),
    balanced_acc_se = se(balanced_acc, na.rm = TRUE),
    tpr_mean = mean(tpr, na.rm = TRUE),
    tpr_se = se(tpr, na.rm = TRUE),
    precision_mean = mean(precision, na.rm = TRUE),
    precision_se = se(precision, na.rm = TRUE),
    MCSF_mean = mean(MCSF, na.rm = TRUE),
    MCSF_se = se(MCSF, na.rm = TRUE),
    Cats_cos_mean = mean(Cats_cos, na.rm = TRUE),
    Cats_cos_se = se(Cats_cos, na.rm = TRUE),
    Tanimoto_scaffold_to_train_mean = mean(Tanimoto_scaffold_to_train, na.rm = TRUE),
    Tanimoto_scaffold_to_train_se = se(Tanimoto_scaffold_to_train, na.rm = TRUE),
  ) %>% ungroup()

calibration_summary$bin = as.numeric(as.character(calibration_summary$bin)) - 1


fig3a = ggplot(calibration_summary, aes(y=balanced_acc_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = balanced_acc_mean - balanced_acc_se, ymax = balanced_acc_mean + balanced_acc_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  coord_cartesian(ylim=c(0.40, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Balanced accuracy', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 

fig3b = ggplot(calibration_summary, aes(y=tpr_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = tpr_mean - tpr_se, ymax = tpr_mean + tpr_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  coord_cartesian(ylim=c(0.4, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Hit rate', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'none',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 

fig3c = ggplot(calibration_summary, aes(y=precision_mean, x=bin, color=reliability_method, fill=reliability_method, linetype=reliability_method))+
  geom_ribbon(aes(ymin = precision_mean - precision_se, ymax = precision_mean + precision_se), size=0, alpha=0.1) +
  geom_line(size=0.35)+
  coord_cartesian(ylim=c(0.40, 1), xlim=c(0, 9))+
  scale_x_continuous( breaks = seq(0, 9, by = 1), labels = as.character(1:10)) +
  labs(y='Precision', x='bins') +
  scale_fill_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_color_manual(values = c('#577788','#97a4ab', '#578d88', '#99beae', '#efc57b','#ef9d43')) +
  scale_linetype_manual(values = c('dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid')) +
  default_theme + theme(legend.position = 'right',
                        plot.margin = unit(c(0.2, 0.2, 0.2, 0.2), "cm")) 


fig3 = plot_grid(fig3a, fig3b, fig3c, ncol=3, labels = c('a', 'b', 'c'), rel_widths = c(1,1,1.8), label_size = 10)
fig3


# save to pdf
pdf('plots/fig3.pdf', width = 180/25.4, height = 45/25.4)
print(fig3)
dev.off()

