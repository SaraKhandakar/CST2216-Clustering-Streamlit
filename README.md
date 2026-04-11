# Customer Segmentation (KMeans) — CST2216 Term Project

## Overview
- This project modularizes the Level-1 notebook `Unsupervised_Clustering_Solution.ipynb`
- It deploys a Streamlit app for customer segmentation using KMeans clustering
- The app helps visualize customer groups based on shopping behavior

## Dataset
- File: `mall_customers.csv`
- Main features used:
  - `Age`
  - `Annual_Income`
  - `Spending_Score`

## Features
- Elbow plot (WCSS)
- Silhouette plot
- KMeans clustering
- 2D or 3D feature visualization
- Cluster summary and insights

## Logging
- Logging is implemented using Python logging module
- Logs are stored in logs/app.log
- Uses file logging and console output
- Tracks data loading, clustering steps, and errors
- Rotating logs prevent large file sizes

## Project Structure
- `app.py`
- `config.py`
- `requirements.txt`
- `runtime.txt`
- `README.md`
- `data/`
  - `mall_customers.csv`
- `src/`
  - `__init__.py`
  - `clustering.py`
  - `data_loader.py`
  - `utils.py`
  - `visuals.py`
- `logs/`
  - `app.log`
- `tests/`
  - `test_smoke.py`

## Run Locally
- Create virtual environment  
  `python -m venv .venv`

- Activate virtual environment  
  `.\.venv\Scripts\activate`

- Install required packages  
  `pip install -r requirements.txt`

- Run the Streamlit app  
  `streamlit run app.py`

## Deployment
- This project is deployed on Streamlit Community Cloud
- The app supports 2D and 3D clustering views
- Optimal number of clusters is explored using Elbow and Silhouette methods

## Links
- GitHub: [CST2216 Clustering Repository](https://github.com/SaraKhandakar/CST2216-Clustering-Streamlit)
- Streamlit: [Live App](https://cst2216-clustering-app-cn3agktndgzhecfzspckwv.streamlit.app/)

## Author
- Shara Khandakar
- Algonquin College
- Business Intelligence Systems Infrastructure