# This file plots the supplementary figure sFig11 and 12 of the paper
#
# WARNING, IC50s are given even if the curve really should not get one. Use the data in the IC50s dataframe instead to label the plots
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


compute_IC50s <- function(df){
  
  IC50s = list()
  for (cpd in unique(df$Compound)){
    print(paste(cpd))
    # cpd='77'
    
    curve_fit <- drm(Activity ~ Dose_molar, 
                     data = subset(df, Compound == cpd),
                     Replicate, #fit separate curves for each timepoint
                     robust = 'mean', #non-robust least squares estimation ("mean")
                     fct = LL.4(names = c("Slope", "Lower", "Upper", "IC50")))
    
    ic50 = data.frame(names = names(curve_fit$coefficients), 
                      coef = curve_fit$coefficients) %>% 
      filter(grepl("^IC50", names)) %>%
      summarize(IC50_M = mean(coef),
                SD_M = sd(coef),
                IC50_µM = mean(coef * 1e6),
                SD_µM = sd(coef * 1e6),
                IC50_nM = mean(coef * 1e9),
                SD_nM = sd(coef * 1e9),
                IC50_log_M = mean(log10(coef)),
                SD_log_M = sd(log10(coef))
      )
    ic50$Compound = cpd
    
    
    # Estimate the IC50 in case there is no lower plateau
    predicted_lower = data.frame(names = names(curve_fit$coefficients), 
                                 coef = curve_fit$coefficients) %>% 
      filter(grepl("^Lower", names))
    
    predicted_upper = data.frame(names = names(curve_fit$coefficients), 
                                 coef = curve_fit$coefficients) %>% 
      filter(grepl("^Upper", names))
    
    measured_upper = subset(df, Compound == cpd)  %>% 
      group_by(Replicate)  %>% 
      summarize(max = max(Activity))
    
    measured_lower = subset(df, Compound == cpd)  %>% 
      group_by(Replicate)  %>% 
      summarize(min = min(Activity))
    
    # if the estimated plateaus are crazy, just assume the max/min measured values
    if (mean(predicted_upper$coef) > mean(measured_upper$max)){
      upper_plateau = measured_upper$max
    } else {
      upper_plateau = predicted_upper$coef
    }
    
    if (mean(predicted_lower$coef) < mean(measured_lower$min)){
      lower_plateau = measured_lower$min
    } else {
      lower_plateau = predicted_lower$coef
    }
    
    # if there is no lower plateau and the curve breaches 50% inhibiton, then the 50% activity point on the curve is the worst case IC50 estimate
    # if there is now lower plateau and the curve does not breach 50%, IC50 cannot be estimated.
    
    if (mean(lower_plateau) < 50){
      
      # halfway_point = mean(upper_plateau - ((upper_plateau - lower_plateau)/2))
      
      doses <- exp(seq(log(min(df$Dose_molar)), log(max(df$Dose_molar)), length.out = 1000))
      Fitted_activity = predict(curve_fit, newdata = data.frame(Dose_molar = doses))
      
      # find the value on the curve that belongs to the halfway point
      ic50_estimate = doses[which.min(abs(Fitted_activity - 50))]
      
      
    } else {
      ic50_estimate = NaN
    }
    
    ic50$ic50_M_worst_case_estimate = ic50_estimate
    ic50$ic50_µM_worst_case_estimate = ic50_estimate * 1e6
    ic50$ic50_nM_worst_case_estimate = ic50_estimate * 1e9
    ic50$ic50_log_M_worst_case_estimate = log10(ic50_estimate)
    
    IC50s[[length(IC50s)+1]] = ic50
  }
  
  IC50s = bind_rows(IC50s)
  
  return(IC50s)
}

curve_fitting <- function(df){
  # Perform inference on curves for plotting
  curve_models = list()
  fitted_curves = list()
  for (cpd_i in 1:length(unique(df$Compound))){
    print(cpd_i)
    cpd = unique(df$Compound)[cpd_i]
    
    curve_fit <- drm(
      Activity ~ Dose_molar,
      data = df %>% filter(Compound == cpd),
      fct = LL.4(names = c("Slope", "Lower", "Upper", "IC50"))
    )
    
    doses <- exp(seq(log(min(df$Dose_molar)), log(max(df$Dose_molar)), length.out = 100))
    preds <- data.frame(
      Compound = cpd,
      Dose_molar = doses,
      Dose_log_molar = log10(doses),
      Fitted_activity = predict(curve_fit, newdata = data.frame(Dose_molar = doses))
    )
    curve_models[[cpd_i]] = curve_fit
    fitted_curves[[cpd_i]] = preds
  }
  names(curve_models) = unique(df$Compound)
  fitted_curves <- bind_rows(fitted_curves)
  
  return(fitted_curves)
  
}

get_ic50_text <- function(ic50s_df, compound){
  
  paste0(
    "pIC50 = ",
    -1*round(subset(IC50s, Compound == compound)$IC50_log_M, 2),
    '±',
    round(subset(IC50s, Compound == compound)$SD_log_M, 2)
  )
}

dose_response_curve_plot <- function(df_curves, df_fitted, ic50s_df, compound_name, protein_name = 'Kinase'){
  
  df_curves_norm_ = subset(df_curves, Compound == compound_name)
  fitted_curves_ = subset(df_fitted, Compound == compound_name)
  
  fig = ggplot() +
    geom_point(data = df_curves_norm_, aes(x = Dose_molar, y = Activity, color = Compound), size = 0.5) +
    geom_errorbar(data = df_curves_norm_, aes(x = Dose_molar, ymin = Activity - SD, ymax = Activity + SD, color = Compound),
                  width = 0.075, size = 0.25) +
    geom_line(data = fitted_curves_, aes(x = Dose_molar, y = Fitted_activity, color = Compound, group = Compound), size = 0.5) +
    scale_color_manual(values = c('#578d88')) +
    scale_x_log10(limit=c(1.5e-09, 1e-05)) +
    scale_y_continuous(limit=c(-5, 150)) +
    annotate("text", x = 1.5e-09, y = 150, 
             label = paste0(get_ic50_text(ic50s_df, compound_name)), hjust = 0, vjust = 1, size = 3) +
    annotation_logticks(scaled = TRUE, sides = 'b') +
    labs(x = paste0('log [', compound_name,'] M'), y = paste0(protein_name,' activity (%)')) +
    default_theme + theme(legend.position = 'none',
                          axis.ticks.x=element_blank())
  
  return(fig)
}


#### Dose-response curves ####

setwd("~/Dropbox/PycharmProjects/JointMolecularModel")

pim1_curves_cpd_A28_A5_A20 <- read_csv('plots/data/hit screening/04_07_25_pim1_cpd_28_05_20_dose_response_long.csv')
pim1_curves_cpd_A23_A10_A2 <- subset(read_csv('plots/data/hit screening/18_07_25_pim1_cpd_23_10_2_dose_response_long.csv'), Type != 'Positive control') # we only use the positive control from one plate

cdk1_curves_cpd_B9_B30_B17 <- read_csv('plots/data/hit screening/21_07_25_cdk1_cpd_9_30_17_dose_response_long.csv')
cdk1_curves_cpd_B19_B8_B3 <- subset(read_csv('plots/data/hit screening/22_07_2025_cdk1_cpd_19_8_3_dose_response_long.csv'), Type != 'Positive control')

df_curves_norm = rbind(process_raw_data(pim1_curves_cpd_A28_A5_A20),
                       process_raw_data(pim1_curves_cpd_A23_A10_A2),
                       process_raw_data(cdk1_curves_cpd_B9_B30_B17),
                       process_raw_data(cdk1_curves_cpd_B19_B8_B3))

# Match intermediate compound IDS used in the lab to the ones I use in the paper/thesis
df_curves_norm$Compound = screening_lookup_table$Cpd_ID[match(df_curves_norm$Compound, screening_lookup_table$Intermediate_cpd_ID)]

# Compute IC50s for all compounds. Even compounds w/o nice plateaus will be assigned an IC50, so take this with a grain of salt.
IC50s = compute_IC50s(df_curves_norm)

IC50s$label_nM = paste0(round(IC50s$IC50_nM), '±', round(IC50s$SD_nM))
IC50s$label_µM = paste0(round(IC50s$IC50_µM, 2), '±', round(IC50s$SD_µM, 2))

# Compute averages over replicates
df_curves_norm = df_curves_norm %>%
  group_by(Protein, Compound, Dose_molar) %>%
  summarise(SD = sd(Activity),
            Activity = mean(Activity))

# Fit curves (i.e. fit a line one the data and do inference over it)
fitted_curves = curve_fitting(df_curves_norm)

sfig11 = plot_grid(dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '4', 'PIM1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '9', 'PIM1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '10', 'PIM1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, 'AZD1208', 'PIM1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '18', 'PIM1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '25', 'PIM1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '29', 'PIM1'),
                   plot_spacer() + default_theme,
                   
                   ncol=4, align = "h", axis = "ltb", 
                   labels = c('a', 'b', 'c', 'g', 
                              'd', 'e', 'f', ''),
                   label_size=10)

# save to pdf
pdf('plots/figures/sfig11.pdf', width = 180/25.4, height = 80/25.4)
print(sfig11)
dev.off()

sfig12 = plot_grid(dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '31', 'CDK1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '33', 'CDK1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '40', 'CDK1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, 'Dinaciclib', 'CDK1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '48', 'CDK1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '53', 'CDK1'),
                   dose_response_curve_plot(df_curves_norm, fitted_curves, ic50s_df, '59', 'CDK1'),
                   plot_spacer() + default_theme,
                   
                   ncol=4, align = "h", axis = "ltb", 
                   labels = c('a', 'b', 'c', 'g', 
                              'd', 'e', 'f', ''),
                   label_size=10)

# save to pdf
pdf('plots/figures/sfig12.pdf', width = 180/25.4, height = 80/25.4)
print(sfig12)
dev.off()
