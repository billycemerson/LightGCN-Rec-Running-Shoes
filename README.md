# Nineten Running Shoe Recommendation System

This project is a practical implementation of a recommender system tailored to Nineten (910) — a local Indonesian running shoe brand. It is designed to help users discover shoes that best fit their preferences based on past customer rating behavior, using collaborative filtering techniques.

## Project Goals

* **Enhance Customer Experience**
  Help runners discover shoes that align with their needs and preferences by suggesting items based on similar users’ behaviors.

* **Support Business Decision-Making**
  Provide insights to Nineten regarding which products tend to co-occur in user preferences, useful for targeted marketing or product bundling.

* **Demonstrate Applied Recommender Modeling**
  Build and showcase a real-world product recommendation pipeline — from raw data to deployable models — using LightGCN and data engineering practices.

## Data Collection

* **Scraped Source**
  Product and user rating data were collected using the **Web Scraper Chrome Extension** from the official Nineten store on Shopee.

* If you're interested in how the scraping was performed, feel free to reach out.

## Project Structure

```
LightGCN-Rec-Running-Shoes/
├── data/
│   ├── scraping/                # Raw data collected using Web Scraper Chrome Extension
│   ├── transform/               # Cleaned & transformed intermediate data
│   └── load/                    # Final dataset for modeling (user_id, product_id, rating, timestamp)
├── results/                     # Saved model outputs and recommendations
├── src/
│   ├── transform.py             # Script to clean and transform raw scraped data
│   ├── load.py                  # Script to prepare model-ready dataset
│   ├── some/other.py            # Utility functions for model, data preprocesing, evaluator, etc.
│   └── LightGCN.ipynb           # Main notebook for training LightGCN model
└── README.md                    # Project documentation
```

## Installation

Clone the repository:

```bash
git clone https://github.com/billycemerson/LightGCN-Rec-Running-Shoes.git
```

## Usage

> ⚠️ **Recommended to use Google Colab** for training the model due to potential GPU requirements.

1. Ensure your `data/transform/` and `data/load/` folders contain the appropriate CSV files (transformed rating data).
2. Open and run `LightGCN.ipynb` to:

   * Load and split data
   * Train the LightGCN model
   * Evaluate the model using top-K metrics
   * Generate sample product recommendations

## Future Improvements

* Integrate product metadata (e.g., shoe type, price, release date)
* Add content-based filtering for cold-start products
* Deploy as an API or lightweight web demo
