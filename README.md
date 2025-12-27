# Air Quality Analysis Project

A comprehensive data science project analyzing air quality data from Yerevan, Armenia using clustering algorithms and machine learning techniques.

## 📊 Project Overview

This project analyzes air quality measurements collected hourly from Yerevan between September and December 2025.  The analysis includes data preprocessing, exploratory data analysis, clustering techniques, and predictive modeling to understand air quality patterns and trends.

## 👥 Contributors

- **Gayane Yemishyan** - [@GayaneYemishyan](https://github.com/GayaneYemishyan)
- **Monika Yepremyan** - [@Monika303](https://github.com/Monika303)
- **Lilit Zalinyan** - [@Lilit862](https://github.com/Lilit862)
- **Viktorya Margaryan** - [@viktoryamargaryan](https://github.com/viktoryamargaryan)

## 📋 Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [License](#license)

## ✨ Features

- **API Integration**: Automated data collection from air quality monitoring stations
- **Data Cleaning**: Comprehensive preprocessing pipeline handling missing values and outliers
- **Feature Engineering**: Cyclical encoding of wind direction and temporal features
- **Exploratory Data Analysis**: Statistical analysis and visualization of air quality patterns
- **Clustering Analysis**: Implementation of multiple clustering algorithms (K-Means, Hierarchical, DBSCAN, GMM)
- **Predictive Modeling**: Machine learning models for air quality prediction
- **Data Visualization**: Interactive plots and charts using matplotlib and seaborn

## 📊 Dataset

### Data Collection
- **Source**: Air Quality API
- **Location**: Yerevan, Armenia
- **Time Period**: September 1, 2025 - December 1, 2025
- **Resolution**:  Hourly measurements
- **Total Records**: 8,777 initial observations → 6,400 cleaned observations

### Measured Parameters
- **Particulate Matter**:  PM1, PM2.5, PM10
- **Meteorological**:  Temperature, Pressure, Humidity
- **Light**: UV Index, Lux (illuminance)
- **Wind**:  Speed and Direction
- **Precipitation**: Rain measurements

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
Google Colab (recommended) or Jupyter Notebook
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

### Clone Repository
```bash
git clone https://github.com/GayaneYemishyan/air-quality-analysis.git
cd air-quality-analysis
```

## 🔬 Methodology

### 1. Data Integration
- Connected to air quality API endpoint
- Specified parameters for Yerevan city
- Retrieved hourly measurements in JSON format
- Saved raw data for reproducibility

### 2. Data Cleaning
```python
# Key cleaning steps: 
- Handled missing values in 'uv' (24. 6%) and 'wind direction' (4.3%)
- Filled missing values with mode
- Removed rows where all PM values (PM1, PM2.5, PM10) were zero
- Verified no duplicate entries
```

### 3. Data Preprocessing

#### Feature Engineering
- **Timestamp Parsing**: Converted to datetime format
- **Temporal Features**: Extracted hour, day of week, day of year, month
- **Wind Direction Encoding**: 
  - Mapped cardinal directions to degrees
  - Applied cyclical encoding using sine and cosine transformations
  ```python
  wind_dir_sin = sin(wind_direction_degrees)
  wind_dir_cos = cos(wind_direction_degrees)
  ```

#### Data Standardization
- Applied StandardScaler to all numerical features
- Ensured zero mean and unit variance
- Prepared data for distance-based algorithms

### 4. Exploratory Data Analysis
- Statistical summaries of all features
- Correlation analysis between variables
- Temporal pattern identification
- Distribution analysis of pollutants

### 5. Clustering Analysis
Implemented multiple clustering algorithms:
- **K-Means**: Partitional clustering for air quality zones
- **Agglomerative Clustering**: Hierarchical approach
- **DBSCAN**:  Density-based outlier detection
- **Gaussian Mixture Models**:  Probabilistic clustering

Evaluation metrics:
- Silhouette Score
- Davies-Bouldin Index

### 6. Predictive Modeling
Machine learning models for air quality prediction:
- **Random Forest Regressor**: Ensemble learning
- **Gradient Boosting Regressor**:  Advanced boosting technique
- **Ridge Regression**: Linear model with L2 regularization

Performance metrics:
- Mean Squared Error (MSE)
- R² Score

## 🛠️ Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**:  Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing

### Visualization
- **matplotlib**:  Static plotting
- **seaborn**: Statistical data visualization

### Data Processing
- **requests**: API data fetching
- **StandardScaler**: Feature normalization
- **PCA**:  Dimensionality reduction

### Machine Learning
- **Clustering**: KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture
- **Regression**: RandomForestRegressor, GradientBoostingRegressor, Ridge

## 📈 Results

### Data Quality
- Successfully cleaned and preprocessed 6,400 observations
- Handled missing values effectively using statistical imputation
- Removed 2,377 invalid records (zero PM readings)

### Feature Distribution
- Temperature range: 0.13°C to 38.12°C (mean: 14.08°C)
- PM2.5 range: 0 to 356 μg/m³ (mean: 25.9 μg/m³)
- Humidity range: 11% to 100% (mean: 52.3%)

## 📝 Usage

### Running the Analysis

1. **Open in Google Colab**:
   ```
   Upload air_quality_analysis.ipynb to Google Colab
   ```

2. **Mount Google Drive** (for data storage):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Execute cells sequentially**:
   - API Integration
   - Data Cleaning
   - Data Preprocessing
   - EDA and Visualization
   - Clustering Analysis
   - Predictive Modeling

### API Configuration

Replace the API endpoint in the notebook: 
```python
url = "YOUR_API_ENDPOINT_HERE"
```

Modify parameters as needed:
```python
params = {
    "city": "Yerevan",
    "start_date": "2025-09-01",
    "end_date": "2025-12-01",
    "resolution": "hourly",
    "format": "json",
    "parameters": ["temperature", "pressure", "humidity", ...]
}
```

## 📊 Data Files

- **[Raw API Data](https://drive.google.com/file/d/1m6p5AfD6LoHMC27QJSWbUr4LP-Y_a5ud/view?usp=sharing)**: Original JSON response from API
- **[Cleaned Data](https://drive.google.com/file/d/1KgUkzqcrCvryDB0fx7xXPgjlNyztE7O8/view?usp=sharing)**: Preprocessed JSON file ready for analysis


## 📧 Contact

For questions or feedback, please reach out to:
- Gayane Yemishyan - [@GayaneYemishyan](https://github.com/GayaneYemishyan)]
  
---

**Note**: This project was developed as part of a Machine Learning course analyzing environmental data from Yerevan, Armenia. 
