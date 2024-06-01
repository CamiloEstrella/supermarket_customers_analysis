import os
print(os.getcwd())
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Change to TkAgg backend
matplotlib.use('TkAgg')

import seaborn as sns
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------------------------------
## 0. Working on Order_Product.csv

try:
    order_product = pd.read_csv("Order_Product.csv", encoding='ISO-8859-1')
    print(order_product.head())
except UnicodeDecodeError:
    print("Failed to read with ISO-8859-1 encoding.")

# -------------------------------------------------------------------------------------------------------
## 1. Load datasets
order = pd.read_csv("Order.csv")
product = pd.read_csv("Product.csv")
aisle = pd.read_csv("Aisle.csv")
department = pd.read_csv("Department.csv")
cruzada = pd.read_csv("cruzada_df.csv")

# -------------------------------------------------------------------------------------------------------
## 2. Extract the First 30 Aisles with the Most Sales

# Join tables to map each product_id to its aisle_id
order_product_info = order_product.merge(product, on = "product_id")

# Counting sales per aisle
aisle_counts = order_product_info.groupby("aisle_id").size().reset_index(name = "sales_count")

# Sort and select the first 30 aisles
top_aisles = aisle_counts.sort_values(by = "sales_count", ascending = False).head(30)

# Relate 'aisle_id' to aisle names
top_aisles = top_aisles.merge(aisle, on = "aisle_id")

# Create histogram
plt.figure(figsize = (12, 8))
sns.barplot(x = 'sales_count', y = 'aisle', data = top_aisles, palette = 'viridis')
plt.xlabel('Number of Sales')
plt.ylabel('Aisle')
plt.title('Aisle Sales')
plt.show()

# -------------------------------------------------------------------------------------------------------
## 3. Reduction of Variables in "cruzada" Table

# Replace spaces with periods in corridor names
top_aisles['aisle'] = top_aisles['aisle'].str.replace(' ', '.')

# Extract aisle names from top_aisles
top_aisle_names = top_aisles['aisle'].tolist()

# Filter 'cruzada' variables to include only those of top_aisle_names
reduced_cruzada = cruzada[top_aisle_names]

# -------------------------------------------------------------------------------------------------------
## 4. Generate Random Samples

sample_sizes = [300, 400, 500, 700, 1000]
samples = {}

for size in sample_sizes:
    samples[size] = reduced_cruzada.sample(n=size, random_state = size)

# -------------------------------------------------------------------------------------------------------
### 5. Validation of Samples

# Function to calculate proportions and create a dataframe
def create_proportions_df(complete_dataset, sample_dataset):
    proportions_complete = complete_dataset.mean()
    proportions_sample = sample_dataset.mean()
    
    proportions_df_complete = pd.DataFrame({
        'Aisle': proportions_complete.index,
        'Proportion': proportions_complete.values,
        'Type': 'Complete'
    })

    proportions_df_sample = pd.DataFrame({
        'Aisle': proportions_sample.index,
        'Proportion': proportions_sample.values,
        'Type': 'Sample'
    })
    
    proportions_df = pd.concat([proportions_df_complete, proportions_df_sample])
    
    return proportions_df

# Function to plot proportions
def plot_proportions(proportions_df, suffix):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Aisle', y='Proportion', hue='Type', data=proportions_df, palette={'Complete': 'blue', 'Sample': 'red'})
    plt.xticks(rotation=90)
    plt.title(f'Comparison of Proportions of Sales between Full and Sample Datasets {suffix}')
    plt.show()

# Calculate and plot proportions for each sample
for size in sample_sizes:
    sample_dataset = samples[size]
    proportions_df = create_proportions_df(reduced_cruzada, sample_dataset)
    plot_proportions(proportions_df, size)

### Step 7: Perform Chi-Square Tests

# Function to perform chi-square test and return adjusted and unadjusted p-values
def perform_chi_square_test(complete_dataset, sample_dataset):
    p_values = []
    
    for column in complete_dataset.columns:
        contingency_table = pd.crosstab(index=complete_dataset[column], columns=sample_dataset[column])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        p_values.append(p)
    
    p_values = np.array(p_values)
    adjusted_p_values = p_values * len(p_values)  # Bonferroni correction
    
    return p_values, adjusted_p_values

# Perform chi-square tests for each sample
for size in sample_sizes:
    sample_dataset = samples[size]
    p_values, adjusted_p_values = perform_chi_square_test(reduced_cruzada, sample_dataset)
    print(f"Results for sample size: {size}")
    print(f"Unadjusted P-values: {p_values}")
    print(f"Adjusted P-values (Bonferroni): {adjusted_p_values}")
    print("---")


### Step 8: Logistic Biplot (Example with PCA)

# For simplicity, let's use PCA as an example, as Python does not have a direct equivalent of `BinaryLogBiplotGD`.


from sklearn.decomposition import PCA

# Apply PCA (as a substitute for BinaryLogBiplotGD)
for size in sample_sizes:
    sample_dataset = samples[size]
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(sample_dataset)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title(f'PCA of Sample Dataset {size}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
