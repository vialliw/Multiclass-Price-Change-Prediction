# Stock Price Change Prediction Using Random Forests

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#Setup-instructions)
- [Usage](#Usage)
- [Visualizations](#Visualizations)
- [Results and Insights](#Results-and-insights)
- [Contributions](#Contributions)
- [License](#License)

## Project Overview

#### Objective:

The primary objective of this project is to predict stock price movements using a Random Forests model. Instead of directly predicting the actual stock price, which is inherently challenging due to its high volatility and sensitivity to external factors, the focus is on predicting price change directions. Predicting actual prices often suffers from drawbacks such as overfitting, high sensitivity to noise, and difficulty in capturing complex market dynamics. By shifting the focus to price change prediction, the model can better generalize patterns and provide actionable insights for traders and investors.


#### Target Classes:

The target variable is defined as the price change between the current day and the 10th day in the future. This price change is categorized into five distinct classes based on the standard deviation of historical price changes:

| Class       | Denote       | Descriptions       |
|----------------|----------------|----------------|
| 2  | Big rise  | Log return of next 10th day greater than 1.5 standard deviation  |
| 1  | Rise  | Log return of next 10th day  between 0.5 and 1.5 standard deviation  |
| 0  | Flat  | Log return of next 10th day  between -0.5 and 0.5 standard deviation  |
| -1  | Drop  | Log return of next 10th day  between -1.5 and -0.5 standard deviation  |
| -2  | Big drop  | Log return of next 10th day less than 1.5 standard deviation  |

This classification approach ensures a balanced representation of market movements and aligns with practical trading strategies.

#### Features:

The model leverages technical indicators as features, which are widely used in financial analysis to identify trends and potential price movements. These indicators include, but are not limited to:

  - [Money Flow Index](https://www.investopedia.com/terms/m/mfi.asp) A day parameter value of 14 was utilized.
  - [Williams %R](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/williams-r) A day parameter value of 14 was utilized.
  - [Rate of Change(ROC)](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc) Both 14-day and intraday ROCs are utilized.
  - [Price/EMA](https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp) 14-day and 40-day EMAs are applied.
  - [Linear Regression Slope](https://trendspider.com/learning-center/linear-regression-slope-a-comprehensive-guide-for-traders/) 14-day and 40-day linear regression slopes are applied.

These features capture historical price and volume patterns, providing a robust foundation for the Random Forests model to learn from.

#### Model and Evaluation:

The Random Forests algorithm is chosen for its ability to handle high-dimensional data, reduce overfitting, and provide interpretable feature importance scores. The model's performance will be evaluated using a confusion matrix, which provides a detailed breakdown of classification accuracy across the five target classes. This evaluation method allows for a clear understanding of the model's strengths and weaknesses in predicting different types of price movements.


#### Expected Outcome:

This project aims to develop a robust and interpretable model for stock price change prediction, offering valuable insights for informed decision-making in financial markets. By focusing on directional changes rather than absolute prices, the model addresses the limitations of traditional price prediction approaches and aligns with practical trading objectives.


## Directory Structure

- src/: Python scripts, including:
  - prediction.py: For sliding window feature generation using 8 technical indicators, training the Random Forest model, evaluating the model and creating visualizations.
- visualizations/: Save generated charts for class distribution and confusion matrix.
- requirements.txt: List of Python dependencies.

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

The Random Forests model was trained and evaluated on the stock price dataset, focusing on predicting price change categories: Big Rise, Rise, Flat, Drop, and Big Drop. Performance was assessed using a normalized confusion matrix, which revealed unexpected results, prompting a reevaluation of the model’s behavior.


#### 1. Model Performance:


Training Accuracy: The model achieved a training accuracy of **99.79%**, suggesting it memorized the training data effectively.
Evaluation Accuracy: Despite the high training accuracy, the evaluation accuracy was **54.21%**, indicating poor generalization to unseen data.
Normalized Confusion Matrix: Surprisingly, the normalized confusion matrix showed diagonal values close to 1.0, implying near-perfect classification for each class. However, this result contradicts the low evaluation accuracy, signaling a critical issue in the evaluation process or data handling.

#### 2. Confusion Matrix Analysis:

The normalized confusion matrix initially appeared promising, with diagonal values near 1.0, suggesting the model correctly classified nearly all instances in each class. However, this result is inconsistent with the overall evaluation accuracy of 54.21%. Possible explanations include:


Data Leakage: There may have been unintentional leakage of future information into the training process, leading to artificially inflated performance metrics.
Incorrect Normalization: The normalization process might have been applied incorrectly, masking true model performance.
Imbalanced Classes: Despite normalization, the model may still be favoring the majority class (e.g., Flat movements) due to class imbalance, leading to misleading diagonal values.


#### 3. Insights and Observations:


Discrepancy in Metrics: The contradiction between the normalized confusion matrix (diagonal values ~1.0) and the low evaluation accuracy (54.21%) highlights a critical issue in the model’s evaluation or data handling. This discrepancy must be resolved before drawing conclusions.
Potential Data Leakage: The near-perfect diagonal values suggest the model may have access to future information, which would render the results invalid for real-world applications.
Class Imbalance: Even with normalization, the model’s performance may still be skewed by the dominance of certain classes (e.g., Flat movements), leading to misleadingly high diagonal values.
Practical Implications: If the model’s true performance aligns with the evaluation accuracy (54.21%), its utility is limited, especially for predicting extreme movements (Big Rise and Big Drop).

#### 4. Recommendations for Improvement:


Investigate Data Leakage: Thoroughly review the data preprocessing pipeline to ensure no future information is inadvertently included in the training process.
Reevaluate Confusion Matrix: Verify the normalization process and consider using raw (unnormalized) metrics or other evaluation methods (e.g., F1-score, precision-recall) to gain a clearer understanding of model performance.
Address Class Imbalance: Apply techniques such as oversampling, undersampling, or class weighting to improve performance on underrepresented classes.
Model Refinement: Simplify the model or apply regularization techniques to reduce overfitting, and consider incorporating additional features or external data to enhance predictive power.

#### Conclusion:

The Random Forests model’s performance is marred by a significant discrepancy between its near-perfect normalized confusion matrix and its low evaluation accuracy. This inconsistency points to potential issues such as data leakage or incorrect evaluation, which must be addressed to ensure reliable results. If the true performance aligns with the evaluation accuracy, the model’s utility is limited, particularly for predicting extreme price movements. Resolving these issues and refining the model will be essential for developing a robust and practical stock price change prediction system.



## Contributions

This project was conceptualized, developed, and executed by Vialli Wong, who played a pivotal role in every stage of the workflow. Below is a breakdown of the contributions:


Project Lead & Developer: Vialli Wong
Designed the project framework and methodology.
Implemented the Random Forests model for stock price change prediction.
Conducted data preprocessing, feature engineering, and model evaluation.
Analyzed results, derived insights, and drafted the project documentation.
[GitHub Profile](https://github.com/vialliw) | [LinkedIn Profile](https://www.linkedin.com/in/kin-chit-vialli-wong-06371094/)

Special recognition is given to the open-source community and libraries that enabled this work, including ta-lib, scikit-learn, pandas, numpy, and matplotlib.


This project is a testament to the dedication and expertise of Vialli Wong, showcasing a blend of technical skill, financial insight, and problem-solving prowess. For inquiries or collaborations, please reach out via the links above.



## License

MIT License

Copyright (c) 2025 Vialli Wong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



