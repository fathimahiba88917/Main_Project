import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD DATA
# =========================================================

customers = pd.read_csv("Customer_500.csv")
products = pd.read_csv("Products_100.csv")
orders = pd.read_csv("orders2500.csv")
order_details = pd.read_csv("orderdetails.csv")


# =========================================================
# 2. CLEAN COLUMN NAMES
# =========================================================

def clean_columns(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


customers = clean_columns(customers)
products = clean_columns(products)
orders = clean_columns(orders)
order_details = clean_columns(order_details)


# =========================================================
# 3. DISPLAY COLUMN NAMES
# =========================================================

print("\n========== COLUMN CHECK ==========")

print("Customers:", customers.columns.tolist())
print("Products:", products.columns.tolist())
print("Orders:", orders.columns.tolist())
print("Order Details:", order_details.columns.tolist())


# =========================================================
# 4. DATA SIZE
# =========================================================

print("\n========== DATA SIZE ==========")

print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)
print("Order Details:", order_details.shape)


# =========================================================
# 5. STANDARDIZE ID COLUMNS
# =========================================================

customers["Customer_ID"] = customers["Customer_ID"].astype(str).str.strip()
orders["Customer_ID"] = orders["Customer_ID"].astype(str).str.strip()

orders["Order_ID"] = orders["Order_ID"].astype(str).str.strip()
order_details["Order_ID"] = order_details["Order_ID"].astype(str).str.strip()

products["Product_ID"] = products["Product_ID"].astype(str).str.strip()
order_details["Product_ID"] = order_details["Product_ID"].astype(str).str.strip()


# =========================================================
# 6. DATA QUALITY CHECK
# =========================================================

print("\n========== DATA QUALITY CHECK ==========")

print("\nMissing values:")
print("Customers:", customers.isnull().sum().sum())
print("Products:", products.isnull().sum().sum())
print("Orders:", orders.isnull().sum().sum())
print("Order Details:", order_details.isnull().sum().sum())


# =========================================================
# 7. MERGE ORDERS + ORDER DETAILS
# =========================================================

sales = orders.merge(
    order_details,
    on="Order_ID",
    how="inner"
)

print("\n========== AFTER ORDERS MERGE ==========")
print("Rows:", len(sales))


# =========================================================
# 8. MERGE WITH PRODUCTS
# =========================================================

sales = sales.merge(
    products,
    on="Product_ID",
    how="inner"
)

print("\n========== AFTER PRODUCTS MERGE ==========")
print("Rows:", len(sales))


# =========================================================
# 9. MERGE WITH CUSTOMERS
# =========================================================

customer_info = customers[
    ["Customer_ID", "Customer_Name", "Gender", "City", "State"]
]

sales = sales.merge(
    customer_info,
    on="Customer_ID",
    how="left"
)

print("\n========== FINAL SALES DATA ==========")
print("Rows:", len(sales))
print("Columns:", len(sales.columns))


# =========================================================
# 10. CALCULATE SALES
# =========================================================

sales["Gross_Sales"] = (
    sales["Price"] * sales["Quantity"]
)

sales["Discount_Amount"] = (
    sales["Gross_Sales"]
    * sales["Discount_Percent"]
    / 100
)

sales["Net_Sales"] = (
    sales["Gross_Sales"]
    - sales["Discount_Amount"]
)

sales["Total_Cost"] = (
    sales["Cost_Price"]
    * sales["Quantity"]
)

sales["Profit"] = (
    sales["Net_Sales"]
    - sales["Total_Cost"]
)


# =========================================================
# 11. KEY BUSINESS METRICS
# =========================================================

total_sales = sales["Net_Sales"].sum()
total_profit = sales["Profit"].sum()
total_orders = sales["Order_ID"].nunique()
total_customers = sales["Customer_ID"].nunique()
total_units = sales["Quantity"].sum()

average_order_value = (
    total_sales / total_orders
)

profit_margin = (
    total_profit / total_sales
) * 100


print("\n========== KEY BUSINESS METRICS ==========")

print(f"Total Sales: ₹{total_sales:,.2f}")
print(f"Total Profit: ₹{total_profit:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Total Customers: {total_customers:,}")
print(f"Total Units Sold: {total_units:,}")
print(f"Average Order Value: ₹{average_order_value:,.2f}")
print(f"Profit Margin: {profit_margin:.2f}%")


# =========================================================
# 12. TOP 10 PRODUCTS
# =========================================================

top_products = (
    sales.groupby("Product_Name")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 PRODUCTS BY SALES ==========")
print(top_products)


# =========================================================
# 13. CATEGORY PERFORMANCE
# =========================================================

category_sales = (
    sales.groupby("Category")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== CATEGORY SALES ==========")
print(category_sales)


# =========================================================
# 14. BRAND PERFORMANCE
# =========================================================

brand_sales = (
    sales.groupby("Brand")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== BRAND SALES ==========")
print(brand_sales)


# =========================================================
# 15. TOP CUSTOMERS
# =========================================================

customer_sales = (
    sales.groupby("Customer_ID")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP 10 CUSTOMERS ==========")
print(customer_sales)


# =========================================================
# 16. STATE PERFORMANCE
# =========================================================

state_sales = (
    sales.groupby("State")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== SALES BY STATE ==========")
print(state_sales)


# =========================================================
# 17. PAYMENT METHOD ANALYSIS
# =========================================================

payment_sales = (
    sales.groupby("Payment_Method")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== SALES BY PAYMENT METHOD ==========")
print(payment_sales)


# =========================================================
# 18. MONTHLY SALES
# =========================================================

orders["Order_Date"] = pd.to_datetime(
    orders["Order_Date"],
    errors="coerce"
)

sales["Order_Date"] = pd.to_datetime(
    sales["Order_Date"],
    errors="coerce"
)

monthly_sales = (
    sales.groupby(
        sales["Order_Date"].dt.to_period("M")
    )["Net_Sales"]
    .sum()
)

print("\n========== MONTHLY SALES ==========")
print(monthly_sales)


# =========================================================
# 19. CHART 1 - CATEGORY SALES
# =========================================================

plt.figure(figsize=(10, 6))

category_sales.head(10).plot(kind="bar")

plt.title("Top Categories by Sales")
plt.xlabel("Category")
plt.ylabel("Net Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("category_sales.png")
plt.show()


# =========================================================
# 20. CHART 2 - TOP PRODUCTS
# =========================================================

plt.figure(figsize=(10, 6))

top_products.sort_values().plot(kind="barh")

plt.title("Top 10 Products by Sales")
plt.xlabel("Net Sales")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig("top_products.png")
plt.show()


# =========================================================
# 21. CHART 3 - STATE SALES
# =========================================================

plt.figure(figsize=(10, 6))

state_sales.head(10).plot(kind="bar")

plt.title("Top States by Sales")
plt.xlabel("State")
plt.ylabel("Net Sales")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("state_sales.png")
plt.show()


# =========================================================
# 22. CHART 4 - MONTHLY SALES
# =========================================================

plt.figure(figsize=(10, 6))

monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Net Sales")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("monthly_sales.png")
plt.show()


# =========================================================
# 23. SAVE FINAL DATA FOR POWER BI
# =========================================================

sales.to_csv(
    "Cosmetics_Final_Sales_Analysis.csv",
    index=False
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n======================================")
print("       PYTHON ANALYSIS COMPLETED")
print("======================================")

print("\nFinal file created:")
print("Cosmetics_Final_Sales_Analysis.csv")

print("\nCharts created:")
print("1. category_sales.png")
print("2. top_products.png")
print("3. state_sales.png")
print("4. monthly_sales.png")

print("\nReady for Power BI!")