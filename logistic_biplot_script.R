## ----setup, include=FALSE-------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

## ---Loading libraries---------------------------------------------------------
library(MultBiplotR)
library(knitr)
library(dplyr)
library(ggplot2)
library(scales)


## ---Loading datasets----------------------------------------------------------
order <- read.csv("Order.csv")
order_product <- read.csv("Order_Product.csv")
product <- read.csv("Product.csv")
aisle <- read.csv("Aisle.csv")
department <- read.csv("Department.csv")

cruzada <- as.data.frame(read.csv("cruzada_df.csv"))


## ---Extract the first 30 aisles with the most sales---------------------------
# Join tables to map each product_id to its aisle_id
order_product_info <- order_product %>%
  inner_join(product, by = "product_id")

# Counting sales per aisle
aisle_counts <- order_product_info %>%
  group_by(aisle_id) %>%
  summarise(sales_count = n()) %>%
  ungroup()

# Sort and select the first 30 aisles
top_aisles <- aisle_counts %>%
  arrange(desc(sales_count)) %>%
  slice(1:30)

# Relate 'aisle_id' to aisle names
top_aisles <- top_aisles %>%
  inner_join(aisle, by = "aisle_id")

# Create histogram
ggplot(top_aisles, aes(x = reorder(aisle, -sales_count), y = sales_count)) +
  geom_bar(stat = "identity", fill = "blue") +
  scale_y_continuous(labels = comma) + # This will change the format of the numbers on the y-axis.
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Aisle", y = "Number of sales", title = "Aisle sales")


## ---Reduction of variables in "cruzada" table-------------------------------------
# Replace spaces with periods in corridor names
top_aisles$aisle <- gsub(" ", ".", top_aisles$aisle)

# Extract aisle names from top_aisles
top_aisle_names <- top_aisles$aisle

# Filter 'cruzada' variables to include only those of top_aisle_names
reduced_cruzada <- cruzada %>%
  select(all_of(top_aisle_names))


## ---Generate a random sample--------------------------------------------------
# 300 sample
set.seed(300)
random_dataset_300 <- reduced_cruzada[sample(nrow(reduced_cruzada), size=300),]

# 400 sample
set.seed(400)
random_dataset_400 <- reduced_cruzada[sample(nrow(reduced_cruzada), size=400),]

# 500 sample
set.seed(500)
random_dataset_500 <- reduced_cruzada[sample(nrow(reduced_cruzada), size=500),]

# 700 sample
set.seed(700)
random_dataset_700 <- reduced_cruzada[sample(nrow(reduced_cruzada), size=700),]

# 1000 sample
set.seed(1000)
random_dataset_1000 <- reduced_cruzada[sample(nrow(reduced_cruzada), size=1000),]


## ---Validation of samples-----------------------------------------------------
# Function to calculate proportions and create a dataframe
create_proportions_df <- function(complete_dataset, sample_dataset, suffix) {
  proportions_complete <- colSums(complete_dataset) / nrow(complete_dataset)
  proportions_sample <- colSums(sample_dataset) / nrow(sample_dataset)
  
  # Create a dataframe with the proportions of the sample and the complete set
  proportions_df <- data.frame(
    Aisle = rep(names(proportions_complete), 2),
    Proportion = c(proportions_complete, proportions_sample),
    Type = rep(c("Complete", "Sample"), each = length(proportions_complete))
  )
  # Convert 'Type' to a factor with consistent levels for the legend
  proportions_df$Type <- factor(proportions_df$Type, levels = c("Complete", "Sample"))
  
  return(proportions_df)
}

# Function to plot proportions
plot_proportions <- function(proportions_df, suffix) {
  ggplot(proportions_df, aes(x = Aisle, y = Proportion, fill = Type)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
    scale_fill_manual(
      values = c("Complete" = "blue", "Sample" = "red"),
      labels = c("Complete dataset", paste("Sample", suffix))
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(y = "Proportion of Sales", x = "Aisles", fill = "Type",
         title = paste("Comparison of Proportions of Sales between Full and Sample Datasets", suffix))
}

# Function to perform chi-square test and return adjusted and unadjusted p-values
perform_chi_square_test <- function(complete_dataset, sample_dataset) {
  successes_complete <- colSums(complete_dataset)
  failures_complete <- nrow(complete_dataset) - successes_complete
  successes_sample <- colSums(sample_dataset)
  failures_sample <- nrow(sample_dataset) - successes_sample
  
  results_chi_square <- sapply(names(successes_complete), function(name) {
    contingency_table <- matrix(c(successes_complete[name], successes_sample[name],
                                   failures_complete[name], failures_sample[name]),
                                 nrow = 2)
    chisq.test(contingency_table)$p.value
  })
  
  adjusted_p_values <- p.adjust(results_chi_square, method = "bonferroni")
  
  # Return a list with adjusted and unadjusted p-values
  return(list(unadjusted_p_values = results_chi_square, adjusted_p_values = adjusted_p_values))
}

# List for storing charts
chart_list <- list()

# Dictionary for storing chi-square test results (adjusted and unadjusted)
results_p_values <- list()

# Process samples, plot and perform chi-square tests
suffixes <- c("300", "400", "500", "700", "1000")

for(suffix in suffixes) {
  sample_dataset <- get(paste("random_dataset_", suffix, sep = ""))
  proportions_df <- create_proportions_df(reduced_cruzada, sample_dataset, suffix)
  chart <- plot_proportions(proportions_df, suffix)
  chart_list[[suffix]] <- chart # Store the chart in the list
  
  results <- perform_chi_square_test(reduced_cruzada, sample_dataset)
  results_p_values[[suffix]] <- results
}

# Print charts
lapply(chart_list, print)

# Print both the unadjusted and adjusted p-values for all samples
lapply(names(results_p_values), function(suffix) {
  results <- results_p_values[[suffix]]
  cat(paste("Results for sample size:", suffix, "\n"))
  
  # Unadjusted P-values
  cat("Unadjusted P-values:\n")
  print(kable(as.data.frame(results$unadjusted_p_values), format = "markdown"))
  
  # Adjusted P-values (Bonferroni)
  cat("\nAdjusted P-values (Bonferroni):\n")
  print(kable(as.data.frame(results$adjusted_p_values), format = "markdown"))
  
  cat("\n---\n")
})


## ---Logistic Biplot-----------------------------------------------------------
# Loop to process each dataset
for(suffix in suffixes) {
  # Construct the name of the current dataset
  dataset_name <- paste("random_dataset_", suffix, sep = "")
  
  # Accessing the dataset using get()
  current_dataset <- get(dataset_name)
  
  # Apply BinaryLogBiplotGD function to current dataset
  newBinaryData <- BinaryLogBiplotGD(current_dataset)
  
  # Generate the plot for the current dataset
  plot(newBinaryData, Mode="ah", ShowAxis = TRUE, margin = 0.4, AbbreviateLabels = FALSE,
       Significant = FALSE, LabelRows = FALSE)
}
