import pandas as pd

# Load phishing and legitimate datasets
phishing_data = pd.read_csv('phishing_data.csv')
legitimate_data = pd.read_csv('legitimate_urls.csv')

# Combine both datasets into one DataFrame
combined_data = pd.concat([phishing_data, legitimate_data])

# Shuffle the dataset to mix phishing and legitimate data
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# Save the combined dataset
combined_data.to_csv('combined_url_data.csv', index=False)

print("Data combined and saved as 'combined_url_data.csv'")
