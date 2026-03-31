import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Bristol-Pink Dashboard (V1.0 MVP)", layout="wide")
st.title("Bristol-Pink Sales Data Analysis (MVP)")
st.markdown("💡 **Current Status**: Data preprocessing engine and historical EDA completed. **Next Phase**: AI prediction module integration.")

uploaded_files = st.file_uploader("Upload Bristol-Pink Sales Data (Select multiple CSVs)", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    try:
        all_data = []
        for file in uploaded_files:
            # Smart cleaning engine: handle different file structures
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
            st.warning("Could not extract valid data. Please check file formats.")
            st.stop()
            
        df = pd.concat(all_data, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        
        st.success(f"Successfully cleaned and merged {len(df)} sales records!")
        
        # --- Sidebar Controls ---
        st.sidebar.header("🔍 Historical Data View")
        start_date = st.sidebar.date_input("Start Date", df['Date'].min())
        end_date = st.sidebar.date_input("End Date", df['Date'].max())
        
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df[mask]
        
        food_data = filtered_df[filtered_df['Category'] == 'Food']
        coffee_data = filtered_df[filtered_df['Category'] == 'Coffee']
        
        top3_foods = food_data.groupby('Product_Name')['Sales_Volume'].sum().nlargest(3).index.tolist()
        top3_coffees = coffee_data.groupby('Product_Name')['Sales_Volume'].sum().nlargest(3).index.tolist()
        
        # --- Core Views ---
        st.header("🏆 Best-Selling Products Fluctuation")
        col1, col2 = st.columns(2)
        with col1:
            df_foods = food_data[food_data['Product_Name'].isin(top3_foods)]
            if not df_foods.empty:
                fig_food = px.line(df_foods, x='Date', y='Sales_Volume', color='Product_Name', title="Top 3 Foods Historical Trend", markers=True)
                st.plotly_chart(fig_food, use_container_width=True)
        with col2:
            df_coffees = coffee_data[coffee_data['Product_Name'].isin(top3_coffees)]
            if not df_coffees.empty:
                fig_coffee = px.line(df_coffees, x='Date', y='Sales_Volume', color='Product_Name', title="Top 3 Coffees Historical Trend", markers=True)
                st.plotly_chart(fig_coffee, use_container_width=True)

        st.info("🚧 **In Progress**: The Machine Learning module (Linear Regression) is currently being trained. The upcoming V2.0 will feature the 4-week sales forecasting charts and algorithm evaluation metrics.")

    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    st.info("Please upload the sales data CSV files to begin analysis.")
