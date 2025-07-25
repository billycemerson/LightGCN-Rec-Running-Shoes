import pandas as pd
import os
import re

# Define input/output paths
scrape_dir = "../data/scraping"
extract_dir = "../data/transform"
os.makedirs(extract_dir, exist_ok=True)

# Define a function to clean and normalize product_name (remove hyphens/slashes)
def normalize_product_name(name):
    name = name.replace('-', ' ').replace('/', ' ').strip()
    name = re.sub(r'\s+', ' ', name)
    return name

# Load and clean the 910_product data
product_df = pd.read_csv(os.path.join(scrape_dir, "910_product.csv"))

# Drop scraping metadata
product_df = product_df.drop(columns=["web-scraper-order", "web-scraper-start-url"])

# Normalize product_name for matching
product_df['product_name'] = product_df['product_name'].apply(normalize_product_name)

# Convert sales strings to integer values
def parse_sales(s):
    s = s.lower().replace(" terjual", "").strip()
    if 'rb' in s:
        return int(float(s.replace('rb', '').replace(',', '.')) * 1000)
    else:
        return int(s)

product_df['product_sales'] = product_df['product_sales'].apply(parse_sales)

# Convert product_price (remove dot separator)
product_df['product_price'] = product_df['product_price'].str.replace('.', '', regex=False).astype(int)

# Round product_rating
product_df['product_rating'] = product_df['product_rating'].astype(float).round().astype(int)

# Clean and extract review data from 3 running types
review_files = ['daily_run.csv', 'recovery_run.csv', 'speed_run.csv']
review_dfs = []

def count_rating_entries(rating_str):
    """
    Count the number of rating elements (each '{}' means 1 star)
    """
    if pd.isna(rating_str) or not isinstance(rating_str, str):
        return None
    return rating_str.count('{')

def clean_review_file(filename):
    print(f"Processing review file: {filename}")
    df = pd.read_csv(os.path.join(scrape_dir, filename))

    # Extract product_id from the product URL
    df['product_id'] = df['web-scraper-start-url'].str.extract(r'\.(\d+)$')

    # Extract and normalize product_name from URL
    df['product_name'] = df['web-scraper-start-url'].str.extract(r'shopee\.co\.id/([^-]+(?:-[^-]+)*?)-i\.\d+')
    df['product_name'] = df['product_name'].str.replace('-', ' ', regex=False)

    # Extract user_id number
    df['user_id'] = df['user_id'].str.extract(r'/shop/(\d+)')
    df = df.dropna(subset=['user_id'])

    # Extract date from timestamp
    df['date'] = pd.to_datetime(df['timestamp'].str.extract(r'^(\d{4}-\d{2}-\d{2})')[0]).dt.date

    # Extract shoe size variation
    df['variasi'] = df['timestamp'].str.extract(r'Variasi:\s*(\d+)')

    # Convert rating list to star count
    df['rating'] = df['rating'].apply(count_rating_entries)

    # Convert 'like' column, treating "Membantu?" as 0
    df['like'] = df['like'].replace('Membantu?', 0).fillna(0).astype(int)

    # Combine 'kategori' and 'reviews' into a single 'review' column
    df['kategori'] = df['kategori'].fillna('')
    df['reviews'] = df['reviews'].fillna('')
    df['review'] = (df['kategori'] + ' ' + df['reviews']).str.strip()

    # Drop unused columns
    df = df.drop(columns=[
        'web-scraper-order',
        'web-scraper-start-url',
        'rating-x',
        'kategori',
        'reviews',
        'timestamp'
    ])

    return df

# Process all 3 review files
for file in review_files:
    cleaned = clean_review_file(file)
    review_dfs.append(cleaned)

# Combine all reviews into one DataFrame
all_reviews_df = pd.concat(review_dfs, ignore_index=True)

# Normalize review product_name for consistent matching
all_reviews_df['product_name'] = all_reviews_df['product_name'].apply(normalize_product_name)

# Merge to assign product_id into product_df based on product_name
merged_df = pd.merge(
    product_df,
    all_reviews_df[['product_name', 'product_id']].drop_duplicates(),
    on='product_name',
    how='left'
)

# Save cleaned 910_product with product_id
merged_df.to_csv(os.path.join(extract_dir, "910_product.csv"), index=False)
print("✅ Saved: 910_product.csv")

# Extract unique user data and save as user_data.csv
user_data_df = all_reviews_df[['user_id', 'user_name']].drop_duplicates(subset='user_id')
user_data_df.to_csv(os.path.join(extract_dir, "user_data.csv"), index=False)
print("✅ Saved: user_data.csv")

# Drop user_name from individual review files and save cleaned versions
for i, (df, file) in enumerate(zip(review_dfs, review_files)):
    df = df.drop(columns=['user_name'])
    output_name = file.replace('.csv', '_clean.csv')
    df.to_csv(os.path.join(extract_dir, output_name), index=False)
    print(f"✅ Saved: {output_name}")