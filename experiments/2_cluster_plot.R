
library(ggtree)
library(ape)
library (readr)
library(treeio)
library(dplyr)

pltcols = c('#c3d6d9',
            '#94d2bd',
            '#0a9396',
            '#e9c46a',
            '#ee9b00',
            '#f4a261',
            '#e76f51',
            '#B43718',
            '#48160A',
            '#86AC86',
            '#588157',
            '#344e41',
            '#213129',
            '#C398BE',
            '#A4659D',
            '#74446E',
            '#4D2E49')


# dataset_name = 'CHEMBL4203_Ki'
# in_path = 'Dropbox/PycharmProjects/JointChemicalModel/results/dataset_clustering/CHEMBL4203_Ki_clustering.csv'
# out_path = 'Dropbox/PycharmProjects/JointChemicalModel/results/dataset_clustering/CHEMBL4203_Ki_clustering.csv'
# plot_path = 'Dropbox/PycharmProjects/JointChemicalModel/results/dataset_clustering/CHEMBL4203_Ki_clustering.csv'

args <- commandArgs(trailingOnly = TRUE)

dataset_name <- args[1]
in_path <- args[2]
out_path <- args[3]
plot_path <- args[4]


# read data
df <- read_csv(in_path, name_repair = 'minimal')
df$scaff_idx = 1:nrow(df)
dist_matrix = as.matrix(df[,1:nrow(df)])
rownames(dist_matrix) = df$scaff_idx

# remove the empty scaffold if there is any. This way we assure that its never
# in the OOD split
empty_idx = which(is.na(df$scaffolds))
if (length(empty_idx) > 0){
  dist_matrix <- dist_matrix[-empty_idx, -empty_idx]
}

# Convert to a distance object
dist_obj <- as.dist(dist_matrix)

# Perform hierarchical clustering using average linkage (UPGMA)
hc <- hclust(dist_obj, method = "complete")

# Convert hclust object to a phylo object
phylo_tree <- as.phylo(hc)

# Cut the tree at the specified height to get 5 clusters
clusters <- cutree(hc, k = 5)

# Add clusters to data and save (make sure its matches with the original scaffold)
clusters_df = rep(NaN, nrow(df))
for (i in 1:length(clusters)){
  scaff_idx = as.numeric(names(clusters)[i])
  clusters_df[scaff_idx] = clusters[i]
}
df$clusters = clusters_df
write.csv(df, out_path, row.names = FALSE)

# Get the number of original (full-molecule) smiles for each cluster of scaffolds
clust_size_original = df %>% group_by(clusters) %>% summarise(across(c("n"), list(sum = sum))) %>% ungroup()
clust_size_original = na.omit(clust_size_original[order(clust_size_original$clusters), ])

# percentage size of each cluster
total_n_scaffolds = length(phylo_tree$tip.label)
total_n_mols = sum(df$n)

cluster_scaffold_sizes = round(table(clusters) * 100 / total_n_scaffolds, 1)
cluster_original_sizes = round(clust_size_original$n_sum * 100 / total_n_mols, 1)


## Plot 


# set tip colour
tip <- get.tree(phylo_tree)$tip.label
tipcol.df <- data.frame(taxa=tip, tipcol=factor(clusters))
levels(tipcol.df$tipcol) = paste0(unique(tipcol.df$tipcol), ": ", cluster_scaffold_sizes, '% scaffolds, ', cluster_original_sizes, '% total')

# set branchcolours
branchColor <- data.frame(node = 1:Nnode2(phylo_tree), colour = '#005f73')

# Open a PDF device
pdf(plot_path, width = 10, height = 6)

# dev.print(pdf, plot_path, width = 8, height = 6)
# plot the data
ggtree(phylo_tree, layout='ape') %<+% branchColor %<+% tipcol.df + aes(colour=I(colour)) +
  geom_tippoint(size=1, alpha=1, aes(color = tipcol)) +
  geom_rootpoint(color="#005f73", size=1) +
  # geom_tiplab(size=2, aes(color = tipcol)) +
  ggplot2::labs(color='Clusters', 
                x=paste0(dataset_name, ' (n = ', nrow(tipcol.df), ' scaffolds)')) +
  theme(plot.margin = unit(c(2, 2, 2, 2), "cm")) +
  scale_color_manual(values=pltcols)

# save plot
# Close the PDF device
dev.off()

# geom_rootpoint(color="black", size=3) +
# WHAT TO DO WITH '' SCAFFOLDS???







