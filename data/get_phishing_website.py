import requests
import pandas as pd

url = 'https://openphish.com/feed.txt'
response = requests.get(url)

# Save the data to a text file
with open('phishing_urls.txt', 'w') as file:
    file.write(response.text)


# Read the text file
with open('phishing_urls.txt', 'r') as file:
    urls = file.readlines()

# Create a DataFrame
df = pd.DataFrame(urls, columns=['url'])

# Add a column for the label (1 for phishing)
df['label'] = 1  # Assuming all URLs from OpenPhish are phishing

# Save to CSV
df.to_csv('phishing_data.csv', index=False)
