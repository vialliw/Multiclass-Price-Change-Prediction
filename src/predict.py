import vw_toolbox as vw
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

def plot_confusion_matrix(historical_df, normalize=True, output_path=''):
    """
    Plot confusion matrix for multi-class classification
    
    Args:
        historical_df: DataFrame containing 'actual_target1' and 'predicted_target1'
        normalize: Whether to normalize the confusion matrix (default: True)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Get unique classes (should be 5 categories)
    classes = sorted(list(set(
        list(historical_df['actual_target1'].dropna().unique()) + 
        list(historical_df['predicted_target1'].dropna().unique())
    )))
    
    # Create labels for the classes
    class_labels = [f'Class {c}' for c in classes]
    
    # Compute confusion matrix
    cm = confusion_matrix(
        historical_df['actual_target1'].dropna(), 
        historical_df['predicted_target1'].dropna(),
        labels=classes
    )
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if len(output_path) > 1:
        plt.savefig(f"{output_path}{os.sep}confusion_matrix.png")
    else:   
        plt.show()
    
    # Also show the non-normalized version if we normalized first
    if normalize:
        cm_counts = confusion_matrix(
            historical_df['actual_target1'].dropna(), 
            historical_df['predicted_target1'].dropna(),
            labels=classes
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels,
                    yticklabels=class_labels)
        plt.title('Confusion Matrix (Counts)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if len(output_path) > 1:
            plt.savefig(f"{output_path}{os.sep}confusion_matrix_counts.png")
        else:   
            plt.show()

def plot_multiclass_distribution(df, column='target1', output_path=''):
    """Plot the distribution of classes in the target column"""
    plt.figure(figsize=(10, 6))
    counts = df[column].value_counts().sort_index()
    counts.plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {column} Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if len(output_path) > 1:
        plt.savefig(f"{output_path}{os.sep}multiclass_distribution.png")
    else:   
        plt.show()

def create_sliding_window_dataset(df, feature_cols, window_size=10):
    """Creates sliding window features from time series data"""
    X = []
    indices = []
    
    for i in range(len(df) - window_size + 1):
        window_data = df.iloc[i:i+window_size][feature_cols].values.flatten()
        X.append(window_data)
        indices.append(i+window_size-1)  # Index of the last element in the window
        
    return np.array(X), indices


def main():
    # Create or load dataset with 5 categories
    print("Loading multi-class dataset...")
    # Initialization
    ticker = sys.argv[1]
    nof_years_data = sys.argv[2]
    output_path = sys.argv[3]
    nof_days_for_target = 10
    short_period, long_period = 14, 40
    # Create or load dataset
    print("Loading dataset...")
    #df = create_sample_data()
    price_df = vw.get_hist_prices(ticker, nof_years_data + "y") 
    price_df = vw.calculate_technical_indicators(ticker, price_df, short_period, long_period)
    price_df = vw.add_numeric_target(ticker, price_df, nof_days_for_target, "LogReturn")
    price_df = vw.add_categorical_target(ticker, price_df, nof_days_for_target, "LogReturn")
    price_df = vw.trim_beginning_nan_rows(price_df, ticker, (long_period - 1))
    price_df = vw.drop_columns(price_df, ticker)
    print(f"Dataset shape: {price_df.shape}")

    # Plot class distribution
    #stats = vw.analyze_price_changes(price_df, (ticker,f"NextDay{nof_days_for_target}LogReturn"), 'Log Return Distribution', 'Log Return (0.0 = No Change)', "c:\\intel\\tmp\\analyze_price_changes" + ticker + '.png')
    plot_multiclass_distribution(price_df, (ticker, "NextDay10LogReturnCtg"), output_path)

    # Define feature columns
    feature_cols = [(ticker, "mfi"), (ticker, "willr"), (ticker, "price_ema_short"), (ticker, "price_ema_long"), (ticker, "lr_slope_short"), (ticker, "lr_slope_long"), (ticker, "roc_short"), (ticker, "roc_intraday")]
    window_size = 10
    
    # Create sliding window dataset
    print(f"Creating sliding window features with window size {window_size}...")
    X, indices = create_sliding_window_dataset(price_df, feature_cols, window_size=window_size)
    y = price_df.iloc[indices][(ticker, "NextDay10LogReturnCtg")].values
    
    print(f"Sliding window X shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Handle NaN values in target (for the last 10 rows)
    y_series = pd.Series(y)
    nan_mask = ~pd.isna(y_series)
    
    X_train, y_train = X[nan_mask], y[nan_mask]
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    
    # Get the unique classes
    classes = sorted(pd.Series(y_train).dropna().unique())
    print(f"Target classes: {classes}")
    
    # Create and train the classification model
    print("Training Random Forest Classification model for multi-class problem...")
    clf_model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=90, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    # Train the model
    clf_model.fit(X_train, y_train)
    print("Model trained successfully!")
    
    # Split data for evaluation
    X_train_eval, X_test, y_train_eval, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train on subset and evaluate
    eval_model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=90, max_depth=12, random_state=42, n_jobs=-1))
    ])
    eval_model.fit(X_train_eval, y_train_eval)
    
    # Evaluate classification model
    y_pred = eval_model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
   
    # Get predictions for all rows
    print("\nGenerating predictions for all data points...")
    all_preds = clf_model.predict(X)
    all_preds_proba = clf_model.predict_proba(X)
    
    # Create a DataFrame with all predictions
    all_dates = price_df.index[indices]
    all_predictions_df = pd.DataFrame({
        'date': all_dates,
        'predicted_target1': all_preds,
        'actual_target1': price_df.iloc[indices][(ticker, "NextDay10LogReturnCtg")].values
    })
    
    # Add columns for probabilities of each class
    for i, c in enumerate(clf_model.classes_):
        all_predictions_df[f'prob_class_{c}'] = all_preds_proba[:, i]
    
    # Add a column to identify which rows have NaN actual values
    all_predictions_df['is_future'] = pd.isna(all_predictions_df['actual_target1'])
    
    # Get the last 10 rows (future predictions)
    future_predictions = all_predictions_df[all_predictions_df['is_future']]
    print("\nPredictions for the last 10 rows (future predictions):")
    print(future_predictions[['date', 'predicted_target1'] + 
                           [f'prob_class_{c}' for c in clf_model.classes_]])
    
    # Get historical predictions (rows with actual values)
    historical_predictions = all_predictions_df[~all_predictions_df['is_future']]
    print(f"\nHistorical predictions (showing last 20 rows):")
    print(historical_predictions.tail(20))
    
    # Calculate performance metrics on historical data
    print("\nPerformance metrics on historical data:")
    hist_accuracy = accuracy_score(
        historical_predictions['actual_target1'], 
        historical_predictions['predicted_target1']
    )
    print(f"Accuracy: {hist_accuracy:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(historical_predictions, True, output_path)
    
    return clf_model, all_predictions_df

if __name__ == "__main__":
    print("Starting multi-class sliding window prediction model...")
    model, predictions = main()
    print("Done!")
