import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Load the combined DataFrame from Google Drive
file_id = '1Wo7cvGe2Pq0TRT-oBP7FosK3KVeJKtvY'
url = f'https://drive.google.com/uc?id={file_id}'
all_predictions_df = pd.read_csv(url)

# Convert the 'date' column to datetime
all_predictions_df['date'] = pd.to_datetime(all_predictions_df['date'])

# Convert to datetime.date for better compatibility with Streamlit slider
all_predictions_df['date'] = all_predictions_df['date'].dt.date

# Calculate residuals
all_predictions_df['residuals'] = all_predictions_df['actual_sales'] - all_predictions_df['predicted_sales']

# Set background style
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #ffffff, #e6e6e6);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333366;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit Dashboard
st.title('Sales Predictions Dashboard')

# Add Tabs
tab1, tab2 = st.tabs(["Dashboard", "Documentation"])

# Dashboard Tab with Heatmap and Graphs
with tab1:
    # Heatmap for total residual error by store and product family
    heatmap_df = all_predictions_df.groupby(['store_nbr', 'family']).agg({'residuals': lambda x: np.abs(x).sum()}).reset_index()
    heatmap_pivot = heatmap_df.pivot(index='store_nbr', columns='family', values='residuals')
    
    st.subheader('Total Residual Error Heatmap')
    fig_heatmap = px.imshow(heatmap_pivot, color_continuous_scale='RdYlGn_r', aspect='auto',
                            labels=dict(x='Product Family', y='Store Number', color='Error'))
    fig_heatmap.update_layout(height=600, width=800)
    st.plotly_chart(fig_heatmap)
    
    # Sidebar for controls
    st.sidebar.header('Filter Options')
    
    # Order stores by highest volume of sales
    store_sales_volume = ['All'] + list(all_predictions_df.groupby('store_nbr')['actual_sales'].sum().sort_values(ascending=False).index)
    store_selection = st.sidebar.selectbox('Select Store Number (Ordered by Sales Volume):', store_sales_volume, index=0, key='store_select')
    
    # Dropdown to select family of product
    family_selection = st.sidebar.selectbox('Select Product Family:', ['All'] + list(all_predictions_df['family'].unique()), index=0, key='family_select')
    
    # Date range selection with datetime.date type
    min_date = all_predictions_df['date'].min()
    max_date = all_predictions_df['date'].max()
    start_date = st.sidebar.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input('End Date', min_value=min_date, max_value=max_date, value=max_date)
    
    # Slider to fine-tune the date range
    date_range = st.sidebar.slider('Select Date Range:',
                                    min_value=min_date,
                                    max_value=max_date,
                                    value=(start_date, end_date))
    
    # Update start and end dates based on the slider
    start_date, end_date = date_range
    
    # Filter data based on selections
    filtered_df = all_predictions_df[(all_predictions_df['date'] >= start_date) & (all_predictions_df['date'] <= end_date)]
    
    if store_selection != 'All':
        filtered_df = filtered_df[filtered_df['store_nbr'] == store_selection]
    
    if family_selection != 'All':
        filtered_df = filtered_df[filtered_df['family'] == family_selection]
    
    # Sum the predictions and sales if 'All' is selected
    if store_selection == 'All' or family_selection == 'All':
        filtered_df = filtered_df.groupby('date', as_index=False).agg({'actual_sales': 'sum', 'predicted_sales': 'sum', 'residuals': 'sum'})
    
    # Separate train and test datasets based on date
    test_start_date = pd.to_datetime('2017-08-15').date()
    train_filtered_df = filtered_df[filtered_df['date'] <= test_start_date]
    test_filtered_df = filtered_df[filtered_df['date'] > test_start_date]
    
    # Predicted vs Actual Sales Graph with Plotly for hover functionality
    st.subheader('Predicted vs Actual Sales Over Time')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_filtered_df['date'],
        y=train_filtered_df['actual_sales'],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Actual Sales: %{y}<extra></extra>',
        legendgroup='group1'
    ))
    fig.add_trace(go.Scatter(
        x=train_filtered_df['date'],
        y=train_filtered_df['predicted_sales'],
        mode='lines+markers',
        name='Predicted Sales (Train)',
        line=dict(color='green'),
        hovertemplate='Date: %{x}<br>Predicted Sales: %{y}<extra></extra>',
        legendgroup='group1'
    ))
    
    # Add test predictions with a different color
    fig.add_trace(go.Scatter(x=test_filtered_df['date'], y=test_filtered_df['predicted_sales'], mode='lines+markers', name='Predicted Sales (Test)', line=dict(color='orange'),
                              hovertemplate='Date: %{x}<br>Predicted Sales (Test): %{y}<extra></extra>'))
    fig.update_layout(
        title=f'Predicted vs Actual Sales for Store {store_selection} and Product Family {family_selection}',
        xaxis_title='Date',
        yaxis_title='Sales',
        xaxis_tickangle=45,
        height=600,
        width=900,
        legend=dict(orientation='h', y=-0.2)  # Place legend at the bottom
    )
    
    filtered_residuals_df = train_filtered_df[train_filtered_df['actual_sales'].notna()]
    
    st.plotly_chart(fig)
    
    # Residuals Graph with Plotly for hover functionality
    st.subheader('Residuals Over Time')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_residuals_df['date'], y=filtered_residuals_df['residuals'], mode='markers', name='Residuals', line=dict(color='red')))
    
    # Convert the date to datetime format for trendline fitting
    filtered_residuals_df['date_ordinal'] = pd.to_datetime(filtered_residuals_df['date']).map(pd.Timestamp.toordinal)
    
    # Add a dotted trend line for the residuals
    trendline = np.polyfit(filtered_residuals_df['date_ordinal'], filtered_residuals_df['residuals'], 1)
    trendline_fn = np.poly1d(trendline)
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=trendline_fn(filtered_residuals_df['date_ordinal']),
        mode='lines',
        name='Residuals Trend Line',
        line=dict(color='blue', dash='dot')
    ))
    
    fig.update_layout(
        title=f'Residuals for Store {store_selection} and Product Family {family_selection}',
        xaxis_title='Date',
        yaxis_title='Residuals',
        xaxis_tickangle=45,
        height=600,
        width=900,
        xaxis=dict(range=[start_date, end_date]),
        legend=dict(orientation='h', y=-0.2)  # Place legend at the bottom
    )
    
    st.plotly_chart(fig)

with tab2:
    st.subheader('Documentation')
    st.markdown("""
    # Store Sales - Time Series Forecasting

## The Dashbaord

    This dashboard provides an interactive way to explore sales predictions and residuals for different stores and product families.
    
    **Heatmap Tab**: Shows the total residual error for each store and product family combination. You can use this to quickly identify areas where the model has larger prediction errors.
    
    **Predicted vs Actual Sales Tab**: Displays the predicted vs actual sales over time, allowing you to compare how well the model performed during both training and test periods.
    
    **Residuals Tab**: Shows the residuals (prediction errors) over time, along with a trend line to visualize any systematic bias in the predictions.
    
    Use the sidebar to filter the data by store, product family, and date range.

## Model Introduction

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
    """)
