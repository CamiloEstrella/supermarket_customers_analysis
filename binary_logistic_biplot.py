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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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

# Function to perform proportional sampling
def proportional_sample(df, n, random_state=None):
    np.random.seed(random_state)
    samples = []
    for col in df.columns:
        col_samples = df[col].value_counts(normalize=True).to_dict()
        col_sampled = np.random.choice(
            list(col_samples.keys()), 
            size=n, 
            p=list(col_samples.values())
        )
        samples.append(pd.Series(col_sampled, name=col))
    return pd.concat(samples, axis=1)

# Generate samples using proportional sampling
sample_sizes = [300, 400, 500, 700, 1000]
random_states = [101, 202, 303, 404, 505]
samples = {}

for size, state in zip(sample_sizes, random_states):
    try:
        samples[size] = proportional_sample(reduced_cruzada, size, random_state=state)
        print(f"Sample size {size} created successfully.")
    except Exception as e:
        print(f"Error creating sample size {size}: {e}")

# Check the samples dictionary
print("Samples dictionary keys:", samples.keys())

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

# Function to perform chi-square test and return adjusted and unadjusted p-values
def perform_chi_square_test(complete_dataset, sample_dataset):
    p_values = []
    
    for column in complete_dataset.columns:
        contingency_table = pd.crosstab(index=complete_dataset[column], columns=sample_dataset[column])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        p_values.append(p)
    
    p_values = np.array(p_values)
    adjusted_p_values = p_values * len(p_values)  # Bonferroni correction
    adjusted_p_values[adjusted_p_values > 1] = 1  # Cap p-values at 1
    
    return p_values, adjusted_p_values

# Validate and plot proportions for each sample
for size in sample_sizes:
    if size in samples:
        sample_dataset = samples[size]
        proportions_df = create_proportions_df(reduced_cruzada, sample_dataset)
        plot_proportions(proportions_df, size)
    else:
        print(f"Sample size {size} is missing in the samples dictionary.")

# -------------------------------------------------------------------------------------------------------
## 5. Perform chi-square tests for each sample

for size in sample_sizes:
    if size in samples:
        sample_dataset = samples[size]
        p_values, adjusted_p_values = perform_chi_square_test(reduced_cruzada, sample_dataset)
        print(f"Results for sample size: {size}")
        print(f"Unadjusted P-values: {p_values}")
        print(f"Adjusted P-values (Bonferroni): {adjusted_p_values}")
        print("---")
    else:
        print(f"Sample size {size} is missing in the samples dictionary.")

# Note: If necessary, the written process is left importing the random samples generated with R.
#########################

# Function to calculate proportions and create a dataframe
# def create_proportions_df(complete_dataset, sample_dataset):
#     proportions_complete = complete_dataset.mean()
#     proportions_sample = sample_dataset.mean()
    
#     proportions_df_complete = pd.DataFrame({
#         'Aisle': proportions_complete.index,
#         'Proportion': proportions_complete.values,
#         'Type': 'Complete'
#     })

#     proportions_df_sample = pd.DataFrame({
#         'Aisle': proportions_sample.index,
#         'Proportion': proportions_sample.values,
#         'Type': 'Sample'
#     })
    
#     proportions_df = pd.concat([proportions_df_complete, proportions_df_sample])
    
#     return proportions_df

# # Function to plot proportions
# def plot_proportions(proportions_df, suffix):
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x='Aisle', y='Proportion', hue='Type', data=proportions_df, palette={'Complete': 'blue', 'Sample': 'red'})
#     plt.xticks(rotation=90)
#     plt.title(f'Comparison of Proportions of Sales between Full and Sample Datasets {suffix}')
#     plt.show()

# # Function to perform chi-square test and return adjusted and unadjusted p-values
# def perform_chi_square_test(complete_dataset, sample_dataset):
#     p_values = []
    
#     for column in complete_dataset.columns:
#         contingency_table = pd.crosstab(index=complete_dataset[column], columns=sample_dataset[column])
#         chi2, p, dof, ex = chi2_contingency(contingency_table)
#         p_values.append(p)
    
#     p_values = np.array(p_values)
#     adjusted_p_values = p_values * len(p_values)  # Bonferroni correction
#     adjusted_p_values[adjusted_p_values > 1] = 1  # Cap p-values at 1
    
#     return p_values, adjusted_p_values

# # Read the full dataset and the samples
# sample_sizes = [300, 400, 500, 700, 1000]
# sample_files = {
#     300: "random_dataset_300.csv",
#     400: "random_dataset_400.csv",
#     500: "random_dataset_500.csv",
#     700: "random_dataset_700.csv",
#     1000: "random_dataset_1000.csv"
# }
# samples = {size: pd.read_csv(file) for size, file in sample_files.items()}

# # Validate and plot proportions for each sample
# for size in sample_sizes:
#     sample_dataset = samples[size]
#     proportions_df = create_proportions_df(reduced_cruzada, sample_dataset)
#     plot_proportions(proportions_df, size)

# # Perform chi-square tests for each sample
# for size in sample_sizes:
#     sample_dataset = samples[size]
#     p_values, adjusted_p_values = perform_chi_square_test(reduced_cruzada, sample_dataset)
#     print(f"Results for sample size: {size}")
#     print(f"Unadjusted P-values: {p_values}")
#     print(f"Adjusted P-values (Bonferroni): {adjusted_p_values}")
#     print("---")

#########################

# -------------------------------------------------------------------------------------------------------
## 6. Binary Logistic Biplot