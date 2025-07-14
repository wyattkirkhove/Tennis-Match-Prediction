# 🎾 Tennis Match Prediction

A machine learning approach to predict tennis match outcomes using comprehensive player statistics, historical performance data, and advanced feature engineering.

##  Project Overview

This project implements a sophisticated tennis match prediction system that analyzes player performance patterns, historical head-to-head records, and various tennis-specific metrics to predict match outcomes. The system uses multiple machine learning models with probability calibration and ensemble methods to achieve robust predictions.

## 🏆 Key Features

### **Comprehensive Feature Engineering**
- **Player Statistics**: Rankings, age, height, Elo ratings (overall and surface-specific)
- **Recent Performance**: Rolling statistics over 5, 10, 20, 50, and 100 match windows
- **Serve & Return Metrics**: First/second serve percentages, break point conversion, ace/double fault rates
- **Surface Performance**: Surface-specific win rates and biases
- **Head-to-Head History**: Historical matchups and win rates
- **Momentum Indicators**: Performance trends and recent form
- **Advanced Metrics**: Serve-return ratios, surface biases, age-elo interactions

### **Machine Learning Pipeline**
- **Multiple Models**: XGBoost, LightGBM, and Logistic Regression
- **Probability Calibration**: Platt scaling and isotonic regression for better probability estimates
- **Ensemble Methods**: Stacking classifier combining all model variants
- **Hyperparameter Optimization**: Pre-tuned parameters for optimal performance
- **Model Persistence**: Saved models for quick inference

## 📊 Data Pipeline

The project follows a comprehensive data processing pipeline:

1. **Raw Data Processing** (`addElo.py`)
   - Loads ATP match data
   - Calculates Elo ratings (overall and surface-specific)
   - Sorts matches chronologically

2. **Feature Engineering** (`featureCreation.py`)
   - Generates rolling performance statistics
   - Creates serve and return metrics
   - Calculates surface-specific features
   - Computes head-to-head statistics

3. **Final Feature Assembly** (`finalFeatures.py`)
   - Combines all engineered features
   - Creates interaction features
   - Prepares final dataset for modeling

4. **Model Training** (`train_models.py`)
   - Trains multiple ML models
   - Applies probability calibration
   - Creates ensemble predictions
   - Evaluates performance metrics

> **Note:**
> The file `data/matches_with_rolling_features.csv` is not included in the repository due to its size. You can regenerate this file by running:
> ```
> python featureCreation.py
> ```
> This script will process the raw data and create all rolling features required for modeling.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone [<repository-url>](https://github.com/wyattkirkhove/Tennis-Match-Prediction)
   cd Tennis-Match-Prediction
   ```
2. **(Recommended) Create a Virtual Environment**

Before installing dependencies, create and activate a virtual environment:

```bash
# Create venv
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# or on macOS/Linux
source venv/bin/activate
```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare the data**
   ```bash
   # Add Elo ratings to match data
   python addElo.py
   
   # Create rolling features and statistics
   python featureCreation.py
   
   # Assemble final feature set
   python finalFeatures.py
   ```

5. **Train the models**
   ```bash
   python train_models.py
   ```
   To retrain all models from scratch, clear the contents of the `models/` directory (except `best_hyperparameters.json`) before running `train_models.py`.

## 📈 Model Performance

The system evaluates models using multiple metrics:

- **Accuracy**: Overall prediction accuracy
- **Brier Score**: Probability calibration quality
- **ROC AUC**: Model discrimination ability

### Model Variants
Each base model (XGBoost, LightGBM, Logistic Regression) is trained with three variants:
- **Base**: Standard model predictions
- **Platt**: Sigmoid calibration for better probabilities
- **Isotonic**: Non-parametric calibration

### Ensemble Approach
A stacking ensemble combines all model variants using logistic regression as the meta-learner, typically achieving the best overall performance.

### Example Results

| Model (Best Variant)         | Train Accuracy | Test Accuracy | Train Brier | Test Brier | Train ROC AUC | Test ROC AUC |
|------------------------------|:-------------:|:-------------:|:-----------:|:----------:|:-------------:|:------------:|
| XGBoost (Isotonic)           | 0.7002        | 0.6717        | 0.1916      | 0.2074     | 0.7800        | 0.7367       |
| Logistic Regression (Base)   | 0.6704        | 0.6730        | 0.2072      | 0.2077     | 0.7371        | 0.7365       |
| LightGBM (Platt)             | 0.6857        | 0.6721        | 0.2002      | 0.2075     | 0.7575        | 0.7371       |
| **Stacking Ensemble**        | 0.6791        | 0.6745        | 0.2013      | 0.2071     | 0.7526        | 0.7379       |

*Other calibration variants performed similarly; see code for full results.*

## 📁 Project Structure

```
Tennis-Match-Prediction/
├── data/                          # Data files
│   ├── matches.csv                # Raw match data
│   ├── mens_matches_sorted_elo.csv # Data with Elo ratings
│   ├── matches_with_rolling_features.csv # Data with rolling stats
│   ├── final_features.csv         # Final feature set
│   └── test_predictions.csv       # Model predictions
├── models/                        # Trained models
│   ├── best_hyperparameters.json  # Optimized parameters
│   ├── preprocessor.joblib        # Feature preprocessing pipeline
│   ├── stacking_model.joblib      # Ensemble model
│   └── [model]_[variant]_model.joblib # Individual models
├── addElo.py                      # Elo rating calculation
├── featureCreation.py             # Rolling feature generation
├── finalFeatures.py               # Final feature assembly
├── train_models.py                # Model training pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Technical Details

### Feature Categories

**Player Demographics**
- Age, height, ranking, ranking points
- Age differences and ranking differentials

**Performance Metrics**
- Recent win rates (5, 10, 20, 50, 100 matches)
- Serve statistics (first/second serve percentages)
- Return performance (break point conversion, return points won)
- Ace and double fault rates

**Advanced Features**
- Elo ratings (overall and surface-specific)
- Surface performance biases
- Head-to-head history and win rates
- Performance trends and momentum indicators
- Serve-return ratios and interaction features

### Model Architecture

1. **Preprocessing Pipeline**
   - Numerical features: Median imputation + Standard scaling (using scikit-learn's SimpleImputer and StandardScaler)
   - Categorical features: Mode imputation + One-hot encoding (using SimpleImputer and OneHotEncoder)

2. **Base Models**
   - XGBoost: Gradient boosting with optimized hyperparameters
   - LightGBM: Light gradient boosting machine
   - Logistic Regression: Linear model baseline

3. **Calibration Methods**
   - Platt Scaling: Sigmoid function calibration (CalibratedClassifierCV with method='sigmoid')
   - Isotonic Regression: Non-parametric calibration (CalibratedClassifierCV with method='isotonic')

4. **Ensemble**
   - Stacking classifier with logistic regression meta-learner (scikit-learn's StackingClassifier)
   - 5-fold cross-validation for meta-learning

## 📊 Usage Examples

### Training Models
```python
# The main training script handles everything automatically
python train_models.py
```

### Loading Trained Models
```python
import joblib

# Load the ensemble model
stacking_model = joblib.load('models/stacking_model.joblib')

# Load the preprocessor
preprocessor = joblib.load('models/preprocessor.joblib')

# Load individual models
xgboost_model = joblib.load('models/xgboost_model.joblib')
```

### Making Predictions
```python
# Preprocess new data
X_processed = preprocessor.transform(new_data)

# Get predictions
probabilities = stacking_model.predict_proba(X_processed)
win_probability = probabilities[:, 1]
```

## 🎯 Key Insights

The model identifies several important factors for tennis match prediction:

1. **Recent Form**: Rolling statistics over the last 5-100 matches capture player momentum
2. **Surface Specialization**: Surface-specific Elo ratings and win rates highlight players' strengths and weaknesses on different surfaces
3. **Head-to-Head History**: Head-to-head historical matchups can provide context for stylistic matchups and in some cases psychological advantages
4. **Serve-Return Balance**: The ratio of serve effectiveness to return performance - This can also providecontext for stylistic matchups
5. **Momentum**: Comparing short-term and long-term trends helps identify players on hot streaks or in slumps

## 🚦 Next Steps / Future Work

While I do not plan to actively maintain or extend this project, here are some logical next steps for anyone interested in building on this work:

- **Update the Dataset:** Incorporate new match data (e.g., 2025 season and beyond) to keep predictions current.
- **Automate Matchup Predictions:** Develop scripts to fetch upcoming matchups and generate predictions using the trained models.
- **Feature Engineering Enhancements:** Explore new features or alternative data sources to further improve model performance.
- **More robust elo calculations:** Vary K-factor for players that take long breaks/have injuries. To learn more about tennis elo, read [Jeff Sackmann's Introduction to Tennis Elo](https://www.tennisabstract.com/blog/2019/12/03/an-introduction-to-tennis-elo/)

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Open-source machine learning libraries (scikit-learn, XGBoost, LightGBM)
- This project uses publicly available tennis match data from [Jeff Sackmann's Tennis Abstract](https://github.com/JeffSackmann/tennis_atp), a well-maintained and respected resource in tennis analytics.

**License:**  
The original datasets are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

> © Jeff Sackmann — Attribution required. Non-commercial use only.  
> See [Jeff Sackmann's GitHub](https://github.com/JeffSackmann) for more details.
---

**Note**: This project is for educational and research purposes. Predictions should not be used for gambling or betting purposes.
