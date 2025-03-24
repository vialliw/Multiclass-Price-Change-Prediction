# Multiclass-Price-Change-Prediction

# Price Change Prediction with Machine Learning

## Table of Contents
- [Project Overview](#project-overview)
- [Features and Target](#features-and-target)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#Setup-instructions)
- [Usage](#Usage)
- [Visualizations](#Visualizations)
- [Results and Insights](#Results-and-insights)
- [Contributions](#Contributions)
- [License](#License)

## Project Overview
This project predicts price change categories based on historical data using a Random Forest model for multiclass classification.

## Features and Target

- **Features:** Derived from 8 technical indicators.
  - [Money Flow Index](https://www.investopedia.com/terms/m/mfi.asp) Day parameter used is 14.
  - [Williams %R](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/williams-r) Day parameter used is 14.
  - [Rate of Change(ROC)](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc) ROC of 14 days and intraday are used.
  - [Price/EMA](https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp) EMA of 14 and 40 days are used.
  - [Linear Regression Slope](https://trendspider.com/learning-center/linear-regression-slope-a-comprehensive-guide-for-traders/) Linear regression slope of 14 and 40 days are used.

- **Target Classes:**

| Class       | Denote       | Descriptions       |
|----------------|----------------|----------------|
| 2  | Big rise  | Log return of next 10th day greater than 1.5 standard deviation  |
| 1  | Rise  | Log return of next 10th day  between 0.5 and 1.5 standard deviation  |
| 0  | Flat  | Log return of next 10th day  between -0.5 and 0.5 standard deviation  |
| -1  | Drop  | Log return of next 10th day  between -1.5 and -0.5 standard deviation  |
| -2  | Big drop  | Log return of next 10th day less than 1.5 standard deviation  |


## Directory Structure

src/: Python scripts, including:
prediction.py: For sliding window feature generation using 8 technical indicators.
model_training.py: For training the Random Forest model.
evaluation.py: For evaluating the model and creating visualizations.
data/: (Optional) Store sample datasets here.
visualizations/: Save generated charts for class distribution and confusion matrix.
requirements.txt: List of Python dependencies.

## Setup-instructions
- Install ta-lib 
  - Download ta-lib wheel file [here](https://github.com/cgohlke/talib-build/releases) according to your python version and platform
  - In terminal, type 'pip install <ta-lib file>'
  - Example: pip install ta_lib-0.6.3-cp312-cp312-win_amd64.whl
- Install required python packages
  - Download [requirements.txt](https://github.com/vialliw/Multiclass-Price-Change-Prediction/blob/main/src/requirements.txt)
  - In terminal, type 'pip install -r requirements.txt'
- Copy [vw_toolbox.py](https://github.com/vialliw/Multiclass-Price-Change-Prediction/blob/main/src/vw_toolbox.py) to your library path

## Usage

- In terminal: Issue the command 'py .\predict.py <ticker_code> <years_of_data> <output_path>'
- Example: py .\predict.py TSLA 4 c:\\windows\\temp

## Visualizations

| Diagram | Thumbnail | Descriptions |
|------------|-------|------|
| **Multiclass Distribution** | <a href="https://raw.githubusercontent.com/vialliw/Multiclass-Price-Change-Prediction/refs/heads/main/images/multiclass_distribution.png" target="_blank"><img src="https://raw.githubusercontent.com/vialliw/Multiclass-Price-Change-Prediction/refs/heads/main/images/multiclass_distribution.png" alt="Multiclass Distribution" title="View Multiclass Distribution" width="250" height="150"></a> | The Multiclass Distribution diagram reveals that fluctuations close to zero occur most frequently. |
| **Standard Confusion Matrix** | <a href="https://raw.githubusercontent.com/vialliw/Multiclass-Price-Change-Prediction/refs/heads/main/images/confusion_matrix_counts.png" target="_blank"><img src="https://raw.githubusercontent.com/vialliw/Multiclass-Price-Change-Prediction/refs/heads/main/images/confusion_matrix_counts.png" alt="Standard Confusion Matrix" title="View Standard Confusion Matrix" width="250" height="200"></a> | The standard confusion matrix effectively demonstrates the model's performance, with the majority of occurrences concentrated along the diagonal cells. This indicates a high degree of accuracy, as these diagonal entries represent correct predictions where the true class matches the predicted class. The minimal off-diagonal values further highlight the model's ability to avoid misclassifications, showcasing its reliability in distinguishing between different classes. |
| **Normalized Confusion Matrix** | <a href="https://raw.githubusercontent.com/vialliw/Multiclass-Price-Change-Prediction/refs/heads/main/images/confusion_matrix.png" target="_blank"><img src="https://raw.githubusercontent.com/vialliw/Multiclass-Price-Change-Prediction/refs/heads/main/images/confusion_matrix.png" alt="Normalized Confusion Matrix" title="View Normalized Confusion Matrix" width="250" height="200"></a> | The normalized multiclass confusion matrix highlights exceptional model performance, with diagonal values approaching 1.0. These values indicate near-perfect accuracy for each class, as they represent the proportion of correctly predicted instances relative to the true class. The minimal off-diagonal values further emphasize the model's robustness, showing negligible misclassifications across all categories. This matrix underscores the model's ability to consistently and accurately classify each class, even in a multiclass setting. |


## Results-and-insights

## Contributions

## License





