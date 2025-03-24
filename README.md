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
- **Target Classes:**
  - `>1.5 stdev`
  - `0.5 to 1.5 stdev`
  - `-0.5 to 0.5 stdev`
  - `-1.5 to -0.5 stdev`
  - `<-1.5 stdev`

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





