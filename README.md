# Multiclass-Price-Change-Prediction

# Price Change Prediction with Machine Learning

## Table of Contents
- [Project Overview](#project-overview)
- [Features and Target](#features-and-target)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Results and Insights](#results-and-insights)
- [Contributions](#contributions)
- [License](#license)

## Project Overview
This project predicts price change categories based on historical data using a Random Forest model for multiclass classification.

## Features and Target
- **Features:** Derived from 8 technical indicators.
  - [Money Flow Index](https://www.investopedia.com/terms/m/mfi.asp)
  - [Williams %R](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/williams-r)
  - [Rate of Change(ROC)](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc)
  - [Price/EMA](https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp)
  - [Linear Regression Slope](https://trendspider.com/learning-center/linear-regression-slope-a-comprehensive-guide-for-traders/)

- **Target Classes:**
  - 2 (Denotes for big rise, log return of next 10th day greater than 1.5 standard deviation)
  - 1 (Denotes for rise, log return of next 10th day  between 0.5 and 1.5 standard deviation)
  - 0 (Denotes for flat, log return of next 10th day  between -0.5 and 0.5 standard deviation)
  - -1 (Denotes for drop, log return of next 10th day  between -1.5 and -0.5 standard deviation)
  - -2 (Denotes for big drop, log return of next 10th day  less than 1.5 standard deviation)

## Directory Structure
src/: Python scripts, including:
data_preparation.py: For sliding window feature generation using 8 technical indicators.
model_training.py: For training the Random Forest model.
evaluation.py: For evaluating the model and creating visualizations.
data/: (Optional) Store sample datasets here.
visualizations/: Save generated charts for class distribution and confusion matrix.
requirements.txt: List of Python dependencies.

## setup-instructions
- Install ta-lib
- Copy vw_toolbox.py to your library path

## usage
Command: py .\talib_rf_predict05d.py <ticker code>
Example: py .\talib_rf_predict05d.py MSFT

## visualizations

## results-and-insights

## contributions

## license





