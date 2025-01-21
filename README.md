# Twitter-Sentiment-Analysis-using-NLP-and-ML

## Project Overview

This project aims to detect hate speech in tweets. A tweet is classified as containing hate speech if it expresses racist or sexist sentiments. The task is to distinguish such tweets from others.

## Dataset Information

- **Objective**: Classify tweets as either containing hate speech (racist/sexist) or not.
- **Labels**: `1` for racist/sexist tweets, `0` for non-racist/sexist tweets.
- **Training Data**: Labeled dataset of 31,962 tweets, provided as a CSV file. Each line contains:
  - Tweet ID
  - Tweet text
  - Corresponding label

## Libraries Used

The project leverages the following libraries:
- `pandas`: Data manipulation and analysis.
- `matplotlib`: Data visualization.
- `seaborn`: Statistical data visualization.
- `scikit-learn`: Machine learning models and evaluation tools.

## Machine Learning Algorithm

The primary algorithm used in this project is Logistic Regression.

### Best Model Accuracy

The best model achieved an accuracy of **95.00%**.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/username/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
    ```
2. Download the dataset and place it in the project directory.

## Usage

1. Preprocess the data:
    ```python
    import pandas as pd
    # Add data preprocessing steps here
    ```
2. Train the model:
    ```python
    from sklearn.linear_model import LogisticRegression
    # Add model training steps here
    ```
3. Evaluate the model:
    ```python
    from sklearn.metrics import f1_score
    # Add model evaluation steps here
    ```

## Results

The Logistic Regression model was trained and tested on the dataset, achieving an accuracy of **95.00%**. Further improvements and experiments can be done to enhance this model.

## License

This project is licensed under the MIT License.
