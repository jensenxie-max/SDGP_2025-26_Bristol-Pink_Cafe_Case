import os
import pandas as pd

def infer_category(filename: str) -> str:
    filename = filename.lower()
    if "coffee" in filename:
        return "Coffee"
    return "Food"

def infer_product_name(filename: str) -> str:
    filename = filename.lower()
    if "croissant" in filename:
        return "Croissant"
    return "Unknown"

def parse_sales_file(file_path: str) -> pd.DataFrame:
    filename = os.path.basename(file_path)
    raw_df = pd.read_csv(file_path)
    print(f"\n=== Raw columns in {filename} ===")
    print(raw_df.columns.tolist())

    # 情况1：Coffee CSV（第二行包含产品名）
    if "coffee" in filename.lower():
        product_names = raw_df.iloc[0, 1:].tolist()  # 取第一行产品名
        data_df = raw_df.iloc[1:].copy()             # 真正数据从第二行开始

        # 重命名列
        new_columns = ["Date"] + product_names
        data_df.columns = new_columns

        # 宽表转长表
        df = data_df.melt(
            id_vars="Date",
            var_name="Product_Name",
            value_name="Sales_Volume"
        )

    # 情况2：Croissant / Food CSV（只有 Date + Number Sold）
    else:
        df = raw_df.copy()
        df.columns = [col.strip() for col in df.columns]

        # 重命名成统一格式
        rename_map = {
            "Date": "Date",
            "Number Sold": "Sales_Volume"
        }
        df = df.rename(columns=rename_map)

        # 补 Product_Name
        df["Product_Name"] = infer_product_name(filename)

        # 调整列顺序
        df = df[["Date", "Product_Name", "Sales_Volume"]]

    # 统一清洗
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Product_Name"] = df["Product_Name"].astype(str).str.strip()
    df["Sales_Volume"] = pd.to_numeric(df["Sales_Volume"], errors="coerce")
    df = df.dropna(subset=["Date", "Product_Name", "Sales_Volume"]).copy()

    return df

def load_uploaded_or_local_data(file_paths) -> pd.DataFrame:
    frames = []
    for path in file_paths:
        current = parse_sales_file(path)
        current["Category"] = infer_category(path)
        frames.append(current)

    if not frames:
        return pd.DataFrame(columns=["Date", "Product_Name", "Category", "Sales_Volume"])

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Category", "Product_Name", "Date"]).reset_index(drop=True)
    return df

def build_daily_series(df: pd.DataFrame, product_name: str) -> pd.Series:
    series = (
        df.loc[df["Product_Name"] == product_name]
        .groupby("Date")["Sales_Volume"]
        .sum()
        .sort_index()
    )
    return series

df = load_uploaded_or_local_data([
    "Pink_CoffeeSales_March_-_Oct_2025.csv",
    "Pink_CroissantSales_March_-_Oct_2025.csv"
])

print("\n=== Data Preview ===")
print(df.head(10))

print("\n=== Category Count ===")
print(df["Category"].value_counts())

print("\n=== Product Names ===")
print(df["Product_Name"].unique())

print("\n=== Americano Daily Series ===")
print(build_daily_series(df, "Americano").head())

print("\n=== Croissant Daily Series ===")
print(build_daily_series(df, "Croissant").head())