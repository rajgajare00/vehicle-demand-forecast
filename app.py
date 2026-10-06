import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vehicle Demand Forecast", page_icon="🚗", layout="wide")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------- load model and data ----------
@st.cache_resource
def load_model():
    return (joblib.load("best_model.pkl"), joblib.load("scaler.pkl"),
            joblib.load("best_model_name.pkl"), joblib.load("columns.pkl"))

@st.cache_data
def load_data():
    df = pd.read_csv("vehicle_sales_demand_dataset.csv", parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    monthly = df.groupby(["Store", "Year", "Month"]).agg(
        Units_Sold=("Units_Sold", "sum"),
        Sales_Records=("Units_Sold", "count"),
        Avg_Price=("Price", "mean"),
        Avg_Discount=("Discount_Percent", "mean"),
        Promotion_Rate=("Promotion", "mean"),
        Avg_Inventory=("Inventory_Level", "mean"),
    ).reset_index()
    monthly["Period"] = pd.to_datetime(dict(year=monthly.Year, month=monthly.Month, day=1))
    return df, monthly

model, scaler, name, columns = load_model()
df, monthly = load_data()
stores = sorted(df["Store"].unique())


def predict(store, month, records, price, discount, promo, inventory):
    row = pd.DataFrame(0, index=[0], columns=columns)
    row["Month"] = month
    row["Sales_Records"] = records
    row["Avg_Price"] = price
    row["Avg_Discount"] = discount
    row["Promotion_Rate"] = promo
    row["Avg_Inventory"] = inventory
    if "Store_" + store in columns:
        row["Store_" + store] = 1
    if name in ["Ridge Regression", "Lasso Regression"]:
        row = scaler.transform(row)
    return max(0, round(float(model.predict(row)[0])))


# ---------- sidebar inputs ----------
st.sidebar.header("Forecast Inputs")
store = st.sidebar.selectbox("Store", stores)
month = st.sidebar.selectbox("Month to forecast", range(1, 13), format_func=lambda m: MONTHS[m - 1])

s = monthly[monthly["Store"] == store]  # defaults come from this store's history
records = st.sidebar.number_input("Expected number of sales records", 1, 100, int(round(s.Sales_Records.mean())), key="r" + store)
price = st.sidebar.number_input("Average Price", 500000.0, 2500000.0, float(round(s.Avg_Price.mean(), -3)), 10000.0, key="p" + store)
discount = st.sidebar.slider("Average Discount %", 0.0, 15.0, float(round(s.Avg_Discount.mean(), 1)), 0.1, key="d" + store)
promo = st.sidebar.slider("Promotion Rate (share of sales on promotion)", 0.0, 1.0, float(round(s.Promotion_Rate.mean(), 2)), 0.05, key="pr" + store)
inventory = st.sidebar.number_input("Average Inventory Level", 10.0, 200.0, float(round(s.Avg_Inventory.mean())), key="i" + store)

# ---------- page ----------
st.title("🚗 Vehicle Units Demand Forecast")
st.caption("Best model: " + name)

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Sales Trends", "ℹ️ Model Info"])

# ===== Tab 1: Forecast =====
with tab1:
    pred = predict(store, month, records, price, discount, promo, inventory)
    avg_units = s.Units_Sold.mean()
    same_month = s[s.Month == month].Units_Sold.mean()
    change = (pred - avg_units) / avg_units * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted units (" + MONTHS[month - 1] + ")", pred, f"{change:+.1f}% vs store average")
    c2.metric("Store average per month", round(avg_units))
    c3.metric("Past average for " + MONTHS[month - 1], round(same_month))

    left, right = st.columns(2)

    # chart 1: store history + forecast line
    with left:
        st.subheader("Monthly sales history - " + store)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(s.Period, s.Units_Sold, marker="o", markersize=3, label="Actual units")
        ax.axhline(pred, color="red", linestyle="--", label="Forecast = " + str(pred))
        ax.axhline(avg_units, color="gray", linestyle=":", label="Average")
        ax.set_ylabel("Units sold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # chart 2: calendar month pattern with forecast highlighted
    with right:
        st.subheader("Average units by month - " + store)
        by_month = s.groupby("Month").Units_Sold.mean().reindex(range(1, 13))
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(MONTHS, by_month.values, color="skyblue", label="Past average")
        ax.bar(MONTHS[month - 1], pred, color="red", width=0.4, label="Forecast")
        ax.set_ylabel("Units sold")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # chart 3: all stores with the same inputs
    st.subheader("Forecast for all stores (same inputs)")
    all_pred = pd.DataFrame({
        "Store": stores,
        "Predicted Units": [predict(st_, month, records, price, discount, promo, inventory) for st_ in stores],
    })
    st.bar_chart(all_pred.set_index("Store"))
    st.dataframe(all_pred, hide_index=True)
    st.download_button("Download forecast as CSV", all_pred.to_csv(index=False), "forecast.csv", "text/csv")

    # what-if: effect of discount
    st.subheader("What-if: effect of discount on forecast")
    whatif = pd.DataFrame({"Discount %": [d for d in range(0, 16)]})
    whatif["Predicted Units"] = [predict(store, month, records, price, d, promo, inventory) for d in whatif["Discount %"]]
    st.line_chart(whatif.set_index("Discount %"))

# ===== Tab 2: Sales trends =====
with tab2:
    st.subheader("Total monthly units - all stores")
    trend = monthly.pivot(index="Period", columns="Store", values="Units_Sold")
    st.line_chart(trend)

    a, b = st.columns(2)
    with a:
        st.subheader("Total units by vehicle")
        st.bar_chart(df.groupby("Vehicle").Units_Sold.sum())
    with b:
        st.subheader("Total units by store")
        st.bar_chart(df.groupby("Store").Units_Sold.sum())

    a, b = st.columns(2)
    with a:
        st.subheader("Avg units per sale: promotion vs no promotion")
        promo_df = df.groupby("Promotion").Units_Sold.mean()
        promo_df.index = ["No promotion", "Promotion"]
        st.bar_chart(promo_df)
    with b:
        st.subheader("Discount % vs units sold")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.scatter(df.Discount_Percent, df.Units_Sold, s=5, alpha=0.4)
        ax.set_xlabel("Discount %")
        ax.set_ylabel("Units sold")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Monthly data (selected store)")
    st.dataframe(s.drop(columns="Period"), hide_index=True)

# ===== Tab 3: Model info =====
with tab3:
    st.write("**Best model:**", name)
    st.write("**Target:** total units sold per store per month")
    st.write("**Input features:**", ", ".join(columns))
    st.write("**Training data:** " + str(len(monthly)) + " store-month records (" + str(len(stores)) + " stores, 36 months)")
    st.write("Models compared: Multiple Linear, Ridge, Lasso, Decision Tree, Random Forest.")
