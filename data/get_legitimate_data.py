import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import deque

def scrape_legitimate_urls(base_urls, target_count=500):
    """
    Scrapes legitimate URLs from specified base URLs and assigns a label of 0 (legitimate).

    Parameters:
        base_urls (list): A list of base URLs to start scraping from.
        target_count (int): Number of legitimate URLs to scrape.

    Returns:
        list: A list of tuples containing (url, label).
    """
    visited_urls = set()
    legitimate_urls = deque()
    urls_to_visit = deque(base_urls)  # Start with the base URLs

    while urls_to_visit and len(legitimate_urls) < target_count:
        current_url = urls_to_visit.popleft()  # Get the next URL to visit

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)

        try:
            response = requests.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']

                # Only consider valid external URLs (not local anchor links)
                if re.match(r'https?://', href):
                    full_url = href if href.startswith('http') else requests.compat.urljoin(current_url, href)
                    if full_url not in visited_urls:
                        legitimate_urls.append((full_url, 0))  # Label as 0 (legitimate)
                        urls_to_visit.append(full_url)

                # Break the loop if we've collected enough legitimate URLs
                if len(legitimate_urls) >= target_count:
                    break

        except requests.RequestException as e:
            print(f"Error fetching {current_url}: {e}")

    return list(legitimate_urls)

def save_to_csv(urls, output_file):
    """
    Saves the list of URLs to a CSV file with the specified format.

    Parameters:
        urls (list): A list of (url, label) tuples to save.
        output_file (str): The output CSV file path.
    """
    df = pd.DataFrame(urls, columns=['url', 'label'])
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Define a list of base URLs to scrape from
    base_urls = [
        "https://www.wikipedia.org", 
        "https://www.google.com", 
        "https://www.amazon.com", 
        "https://www.bbc.com", 
        "https://www.reddit.com", 
        "https://www.stackoverflow.com", 
        "https://www.linkedin.com", 
        "https://www.github.com"
    ]
    
    output_file = "legitimate_urls.csv"

    print("Scraping legitimate URLs...")
    legitimate_urls = scrape_legitimate_urls(base_urls, target_count=500)  # Collecting at least 500 URLs
    print(f"Found {len(legitimate_urls)} legitimate URLs.")
    
    save_to_csv(legitimate_urls, output_file)
    print(f"Legitimate URLs saved to {output_file}.")
