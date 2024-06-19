
library(ggtree)
library(ape)
library (readr)
library(treeio)


# in_path = 'Dropbox/PycharmProjects/JointChemicalModel/CHEMBL4203_Ki_clustering.csv'
# out_path = 'Dropbox/PycharmProjects/JointChemicalModel/CHEMBL4203_Ki_clustering.csv'
# plot_path = 'Dropbox/PycharmProjects/JointChemicalModel/CHEMBL4203_Ki_clustering.pdf'

args <- commandArgs(trailingOnly = TRUE)

dataset_name <- args[1]
in_path <- args[2]
out_path <- args[3]
plot_path <- args[4]


# read data
df <- read_csv(in_path, name_repair = 'minimal')
dist_matrix = df[,1:(ncol(df)-3)]


# Convert to a distance object
dist_obj <- as.dist(dist_matrix)

# Perform hierarchical clustering using average linkage (UPGMA)
hc <- hclust(dist_obj, method = "complete")

# Convert hclust object to a phylo object
phylo_tree <- as.phylo(hc)

# Specify the height at which you want to color branches
cut_height <- hc$height[length(hc$height) - 4]

# Cut the tree at the specified height to get the clusters
clusters <- cutree(hc, h = cut_height)

# Add clusters to data and save
df$clusters = clusters
write.csv(df, out_path, row.names = FALSE)

## Plot 

# percentage size of each cluster
cluster_sizes = round(table(clusters) * 100 / length(phylo_tree$tip.label), 2)

# set tip colour
tip <- get.tree(phylo_tree)$tip.label
tipcol.df <- data.frame(taxa=tip, tipcol=factor(clusters))
levels(tipcol.df$tipcol) = paste0(tipcol.df$tipcol, ": ", cluster_sizes, '%')

# set branchcolours
branchColor <- data.frame(node = 1:Nnode2(phylo_tree), colour = '#005f73')

# Open a PDF device
pdf(plot_path, width = 8, height = 6)
# dev.print(pdf, plot_path, width = 8, height = 6)
# plot the data
ggtree(phylo_tree, layout='daylight') %<+% branchColor %<+% tipcol.df + aes(colour=I(colour)) +
  geom_tippoint(size=1, alpha=1, aes(color = tipcol)) +
  # geom_tiplab(size=2, aes(color = tipcol)) +
  ggplot2::labs(color='Clusters', 
                x=paste0(dataset_name, ' (n = ', nrow(tipcol.df), ' scaffolds)')) +
  theme(plot.margin = unit(c(2, 2, 2, 2), "cm")) +
  scale_color_manual(values= c("#94d2bd", "#0a9396", "#ee9b00", "#c3d6d9", "#f8cd48"))

# save plot
# Close the PDF device
dev.off()


# WHAT TO DO WITH '' SCAFFOLDS???







