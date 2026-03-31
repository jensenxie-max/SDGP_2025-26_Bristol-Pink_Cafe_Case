import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Bristol-Pink Dashboard", layout="wide")
st.title("Bristol-Pink Bakery Sales Prediction Dashboard")

# File upload component
uploaded_files = st.file_uploader("Upload Bristol-Pink Sales Data (Select multiple CSVs)", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    try:
        all_data = []
        for file in uploaded_files:
            filename_lower = file.name.lower()
            category = 'Coffee' if 'coffee' in filename_lower else 'Food'
            
            df_raw = pd.read_csv(file)
            
            if pd.isna(df_raw.iloc[0, 0]):
                file.seek(0)
                df_temp = pd.read_csv(file, skiprows=1)
                df_temp.rename(columns={df_temp.columns[0]: 'Date'}, inplace=True)
                for col in df_temp.columns[1:]:
                    temp_df = df_temp[['Date', col]].copy()
                    temp_df.columns = ['Date', 'Sales_Volume']
                    temp_df['Product_Name'] = col
                    temp_df['Category'] = category
                    all_data.append(temp_df)
            else:
                if 'Date' in df_raw.columns and 'Number Sold' in df_raw.columns:
                    parts = file.name.split('_')
                    product_name = parts[1].replace('Sales', '') if len(parts) > 1 else 'Unknown'
                    temp_df = df_raw[['Date', 'Number Sold']].copy()
                    temp_df.columns = ['Date', 'Sales_Volume']
                    temp_df['Product_Name'] = product_name
                    temp_df['Category'] = category
                    all_data.append(temp_df)

        if not all_data:
            st.error("Could not extract valid data. Please check your files.")
            st.stop()
            
        df = pd.concat(all_data, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        
        # --- Sidebar Controls ---
        st.sidebar.header("⚙️ Prediction Settings")
        training_weeks = st.sidebar.slider("Training Period (Weeks)", min_value=4, max_value=8, value=4)
        
        st.sidebar.header("🔍 Historical Data Zoom")
        start_date = st.sidebar.date_input("Start Date", df['Date'].min())
        end_date = st.sidebar.date_input("End Date", df['Date'].max())
        
        # --- Dashboard Tabs ---
        tab1, tab2, tab3 = st.tabs(["📊 Top 3 Historical Analysis", "📈 4-Week Sales Forecast", "⚙️ Algorithm Evaluation"])
        
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df[mask]
        
        food_data = filtered_df[filtered_df['Category'] == 'Food']
        coffee_data = filtered_df[filtered_df['Category'] == 'Coffee']
        
        top3_foods = food_data.groupby('Product_Name')['Sales_Volume'].sum().nlargest(3).index.tolist()
        top3_coffees = coffee_data.groupby('Product_Name')['Sales_Volume'].sum().nlargest(3).index.tolist()
        
        with tab1:
            st.header("🏆 Best-Selling Products Fluctuation")
            col1, col2 = st.columns(2)
            with col1:
                df_foods = food_data[food_data['Product_Name'].isin(top3_foods)]
                if not df_foods.empty:
                    fig_food = px.line(df_foods, x='Date', y='Sales_Volume', color='Product_Name', title="Top 3 Foods Fluctuation", markers=True)
                    st.plotly_chart(fig_food, use_container_width=True)
            with col2:
                df_coffees = coffee_data[coffee_data['Product_Name'].isin(top3_coffees)]
                if not df_coffees.empty:
                    fig_coffee = px.line(df_coffees, x='Date', y='Sales_Volume', color='Product_Name', title="Top 3 Coffees Fluctuation", markers=True)
                    st.plotly_chart(fig_coffee, use_container_width=True)

        with tab2:
            st.header(f"🔮 4-Week Forecast (Based on {training_weeks}-Week Training Data)")
            top_products = top3_foods + top3_coffees
            
            if top_products:
                selected_product = st.selectbox("👉 Select a product to forecast", top_products)
                
                product_df = df[df['Product_Name'] == selected_product].copy()
                product_df = product_df.sort_values('Date')
                
                train_days = training_weeks * 7
                train_df = product_df.tail(train_days).copy()
                train_df['Day_Index'] = np.arange(len(train_df))
                
                if len(train_df) > 0:
                    X_train = train_df[['Day_Index']]
                    y_train = train_df['Sales_Volume']
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    future_days = 28
                    future_indices = np.arange(len(train_df), len(train_df) + future_days).reshape(-1, 1)
                    predictions = model.predict(future_indices)
                    future_dates = pd.date_range(start=train_df['Date'].max() + pd.Timedelta(days=1), periods=future_days)
                    
                    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Sales': np.round(predictions, 1)})
                    
                    sub_tab1, sub_tab2 = st.tabs(["📉 Graphical View", "🗂️ Tabular View"])
                    with sub_tab1:
                        fig_pred = px.line(pred_df, x='Date', y='Predicted_Sales', title=f"Projected Sales Trend: {selected_product}", markers=True)
                        fig_pred.update_traces(line_color='red')
                        st.plotly_chart(fig_pred, use_container_width=True)
                    with sub_tab2:
                        st.dataframe(pred_df)
            
        with tab3:
            st.header("⚖️ Algorithm Accuracy Comparison (Optional View)")
            st.markdown("The metrics below reflect the error margins of the current algorithm applied to the historical training data. **Adjust the training period slider on the left to observe how the error metrics change.**")
            if 'model' in locals() and len(train_df) > 0:
                y_pred_train = model.predict(X_train)
                mse = mean_squared_error(y_train, y_pred_train)
                col_a, col_b = st.columns(2)
                col_a.metric(label="Mean Squared Error (MSE)", value=np.round(mse, 2))
                col_b.metric(label="Root Mean Squared Error (RMSE)", value=np.round(np.sqrt(mse), 2))

    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}")
else:
    st.info("Please click 'Browse files' and hold Ctrl/Cmd to select all relevant CSV files for upload.")
