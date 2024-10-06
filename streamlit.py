import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components

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

# Sidebar for controls
st.sidebar.header('Filter Options')

# Order stores by highest volume of sales
store_sales_volume = ['All'] + list(all_predictions_df.groupby('store_nbr')['actual_sales'].sum().sort_values(ascending=False).index)
store_selection = st.sidebar.selectbox('Select Store Number (Ordered by Sales Volume):', store_sales_volume, index=0)

# Dropdown to select family of product
family_selection = st.sidebar.selectbox('Select Product Family:', ['All'] + list(all_predictions_df['family'].unique()), index=0)

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

# Predicted vs Actual Sales Graph with Plotly for hover functionality
st.subheader('Predicted vs Actual Sales Over Time')
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df['date'],
    y=filtered_df['actual_sales'],
    mode='lines+markers',
    name='Actual Sales',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Actual Sales: %{y}<extra></extra>',
    legendgroup='group1'
))
fig.add_trace(go.Scatter(
    x=filtered_df['date'],
    y=filtered_df['predicted_sales'],
    mode='lines+markers',
    name='Predicted Sales (Train)',
    line=dict(color='green'),
    hovertemplate='Date: %{x}<br>Predicted Sales: %{y}<extra></extra>',
    legendgroup='group1'
))

# Add test predictions with a different color
fig.add_trace(go.Scatter(x=filtered_df[filtered_df['actual_sales'].isna()]['date'], y=filtered_df[filtered_df['actual_sales'].isna()]['predicted_sales'], mode='lines+markers', name='Predicted Sales (Test)', line=dict(color='orange'),
                          hovertemplate='Date: %{x}<br>Predicted Sales (Test): %{y}<extra></extra>'))
fig.update_layout(
    title=f'Predicted vs Actual Sales for Store {store_selection} and Product Family {family_selection}',
    xaxis_title='Date',
    yaxis_title='Sales',
    xaxis_tickangle=45,
    height=600,
    width=900,
    xaxis=dict(range=[start_date, end_date]),
    legend=dict(orientation='h', y=-0.2)  # Place legend at the bottom
)

filtered_residuals_df = filtered_df[filtered_df['actual_sales'].notna()]

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
