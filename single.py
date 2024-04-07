import pandas as pd

# Load the dataset
data = pd.read_csv("Data.csv")

"""
Bootstrap sampling is used in this case to perform data augmentation by generating synthetic data 
that resembles the original dataset(Table 2). This technique is useful when we have a limited amount of data and want to 
increase the size of the dataset for training machine learning models without collecting additional data.
Bootstrap sampling involves randomly selecting rows from the dataset with replacement until 
we have the desired number of samples(50,000). We use a while loop to achieve this.
"""

# Function to perform bootstrap sampling
def bootstrap_sampling(data, num_samples):
    synthetic_data = []
    n = len(data)
    while len(synthetic_data) < num_samples:
        sample = data.sample(n=1, replace=True)
        synthetic_data.append(sample)

    return pd.concat(synthetic_data, ignore_index=True)

# Generate synthetic data using bootstrapping
synthetic_data = bootstrap_sampling(data, 50000)

# Print the number of desired agents i.e 50k in the synthetic dataset
print("Number of desired agents:", len(synthetic_data))

# Save the synthetic dataset to a CSV file
synthetic_data.to_csv("Synthetic_Data_Bootstrapping.csv", index=False)

# Compute and save the frequencies
frequencies = synthetic_data.apply(pd.value_counts).fillna(0)
print("Frequencies:")
print(frequencies)

# Save the frequencies into a text file
with open("Synthetic_Frequencies.txt", "w") as file:
    file.write(frequencies.to_string())

#This is a validation steps where you can compare proportions between Table 2 and synthetic_data. 
# Compute proportions of each category in the sample dataset
sex_proportions = data['Sex'].value_counts(normalize=True)
age_group_proportions = data['Age_category'].value_counts(normalize=True)
education_level_proportions = data['Highest_education_level'].value_counts(normalize=True)
print(sex_proportions)
print(age_group_proportions)
print(education_level_proportions)


# Read the synthetic dataset(output)
data = pd.read_csv("Synthetic_Data_Bootstrapping.csv")

# Compute proportions of each category in the synthetic dataset 
#you can compare both original and synthetic proportions for validation.
sex_proportions = data['Sex'].value_counts(normalize=True)
age_group_proportions = data['Age_category'].value_counts(normalize=True)
education_level_proportions = data['Highest_education_level'].value_counts(normalize=True)
print(sex_proportions)
print(age_group_proportions)
print(education_level_proportions)

#printed to check if we had 50,000 rows.
len(synthetic_data)



