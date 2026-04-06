# Customer Segmentation (KMeans) - CST2216 Term Project

## Overview
- This project modularizes the Level-1 notebook `Unsupervised_Clustering_Solution.ipynb`
- It builds a Streamlit app for customer segmentation using KMeans clustering
- The app helps visualize customer groups based on shopping behavior

## Dataset
- File: `mall_customers.csv`
- Dataset contains customer information used for clustering
- Typical features include:
  - Age
  - Annual Income
  - Spending Score

## Features
- Elbow plot (WCSS)
- Silhouette plot
- KMeans clustering
- 2D or 3D feature visualization
- Cluster summary and insights

## Project Structure
- `app.py`
- `config.py`
- `requirements.txt`
- `runtime.txt`
- `README.md`
- `data/`
  - `mall_customers.csv`
- `logs/`
  - `app.log`
- `models/`
  - `__init__.py`
  - `clustering.py`
  - `data_loader.py`
  - `utils.py`
  - `visuals.py`
- `tests/`
  - `test_smoke.py`

## Run Locally
- Create virtual environment
- `python -m venv .venv`

- Activate virtual environment
- `.\.venv\Scripts\activate`

- Install required packages
- `pip install -r requirements.txt`

- Run the Streamlit app
- `streamlit run app.py`

## Deployment
- This project is deployed on Streamlit Community Cloud

## Links
- GitHub: [CST2216 Clustering Repository](https://github.com/SaraKhandakar/CST2216-Clustering-Streamlit)
- Streamlit: [Live App](https://cst2216-clustering-app-cn3agktndgzhecfzspckwv.streamlit.app/)

## Author
- Shara Khandakar
- Algonquin College
- Business Intelligence Systems Infrastructure