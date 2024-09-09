# Store Sales - Time Series Forecasting

## Introduction

This project was part of the [Kaggle "Store Sales - Time Series Forecasting" competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview), where the goal was to predict store sales using historical sales data. The challenge involved forecasting sales at different stores for various product families, and the objective was to minimize the root mean squared logarithmic error (RMSLE).

The training data spanned from the start of 2013 until August 15, 2017, and predictions were required for the period from August 16 to August 30, 2017, covering 15 days.

In this project, I used five different models, improving the data processing and model structure in each iteration. Below is a detailed explanation of my approach and the results achieved.

## Data Preprocessing

The initial data preprocessing steps were largely consistent across the models:

- **Handling missing values**: Simple imputation was used for any missing data.
- **Feature engineering**: I created time-based such as `year`, `month` and `day` and lag features such as sales with 30 days and 365 days lag. External data holiday events were also incorporated.
- **Splitting data**: I used time-based splits to ensure the model could generalize on future data.

Key changes and improvements made during the modeling process:

- **Model 2**: I discovered that some stores had zero sales for extended periods, and removing these data points improved model performance.
- **Model 3**: I introduced a critical feature that represented the average sales for each day of the week (Sunday to Saturday) for each specific store and product family. This feature resulted in the largest improvement in score.

## Modeling

### Choice of XGBoost and Random Forest

I opted to use **XGBoost** and **Random Forest** over other models for the following reasons:

- **XGBoost** is well-suited for structured data and provides excellent performance due to its ability to handle missing values, automatic feature importance calculation, and robustness in generalizing on complex patterns. It also supports parallelized training, making it efficient for large datasets.
- **Random Forest** was chosen as it tends to perform well on datasets with high variance and can prevent overfitting by averaging the results of multiple decision trees. It is easy to interpret and provides a good balance of accuracy and speed without requiring extensive hyperparameter tuning.

Both models are also well-regarded for their performance in Kaggle competitions, particularly in time-series and tabular data challenges like this one, making them ideal candidates.

### Combined Models (Models 1 to 3)

In the first three iterations, I used a single combined model to predict the sales for each store and product family:

- **Model 1**: XGBoost with grid search for hyperparameter tuning. The focus was on tuning tree-based parameters.
- **Model 2**: Random Forest, where I removed data points from stores with zero sales for long periods.
- **Model 3**: I added the average sales feature for each day of the week, which significantly improved performance.

### Store-Type-Specific Models (Model 4)

For the fourth model, I realized that stores behaved differently based on their type (A to E). After analyzing this, I created separate models for each store type:
![image](https://github.com/user-attachments/assets/92920639-745e-4542-912c-7769a4dd403b)

![image](https://github.com/user-attachments/assets/210192f7-9a9a-48da-b99c-384fb1aa4fc9)

- Store types B and D were combined as they exhibited similar behavior.
- I also excluded data before 2016 due to a drastic change in oil prices, which improved predictions.

### Ensemble Model (Model 5)

In the final model, I used a combination of Random Forest and XGBoost predictions to create an ensemble model. This approach resulted in the best overall performance.

## Results

| Submission | Rank | Score (RMSLE) |
|------------|------|---------------|
| Model 1    | 413  | 0.7091        |
| Model 2    | 412  | 0.68513       |
| Model 3    | 131  | 0.5045        |
| Model 4    | 105  | 0.4735        |
| Model 5    | 98   | 0.4587        |

The most significant improvement occurred in Model 3 when I added the average sales by day feature. Additionally, creating store-type-specific models and combining predictions in Model 5 further boosted performance.

## Conclusion

The iterative approach in this project, from basic combined models to specialized models for store types, and the introduction of key features like average sales by day, helped achieve substantial improvements in performance. Future work could focus on integrating more sophisticated time-series models or further refining the ensemble approach.

---

