# 📦 Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from io import BytesIO


def fig1_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


# 📊 Set Page Configuration
st.set_page_config(page_title="NYC Collision Analysis", layout="wide")

# 🏁 Page Title
st.title("🚦 NYC Vehicle Collision Insights & Injury Prediction")
st.write("""
Analyze vehicle collisions in New York City — visualize trends, understand patterns, and predict injury or fatality outcomes using machine learning.
""")

# 🔍 Sidebar Navigation
with st.sidebar:
    st.title("📌 Navigation")
    section = st.radio("Go to Section", [
        "Introduction",
        "Dataset Overview",
        "EDA - Part 1: Temporal & Categorical",
        "EDA - Part 2: Severity & Contributing Factors",
        "Geospatial Visualization",
        "Predictions",
    ])


# ✅ Load Datasets (Cached for performance)
@st.cache_data
def load_data():
    df_raw = pd.read_csv("./data/raw_data/nyc_collisions_2024_to_2020.csv",nrows=50)
    df_cleaned = pd.read_csv("./data/cleaned_data/nyc_collisions_cleaned.csv")
    return df_raw, df_cleaned

# 📦 Load once
df_raw, df_cleaned = load_data()

if section == "Introduction":
    st.title("📌 Project Introduction – NYC Vehicle Collisions")

    st.markdown("""
    ### 🚨 Why This Project Matters

    Motor vehicle collisions are a major public safety concern in New York City.  
    These incidents frequently result in injuries or fatalities, affecting thousands of pedestrians, cyclists, and motorists each year.

    According to the [**NYPD Open Data Portal**](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data), tens of thousands of crash incidents are recorded annually — ranging from minor collisions to severe crashes involving multiple injuries or deaths.

    Understanding the **where, when, and why** behind these collisions can play a crucial role in improving safety measures, identifying hotspots, and guiding policy.

    ---

     ### 🎯 Project Goal

    This project analyzes and models five years (2020-2024) of NYC collision data to provide actionable insights and real-time predictions. The key components include:

    - ✅ **Data Cleaning & Feature Engineering**: Convert raw data into meaningful patterns, extract time and location features, and encode relevant fields.
    - 📊 **Exploratory and Geospatial Analysis**: Reveal patterns across boroughs, hours, weekdays, and contributing factors using visualizations and maps.
    - 🤖 **Predictive Modeling**:
        - **Injury Count Regression**: Estimates how many individuals are likely to be injured in a crash.
        - **Injury Classification**: Determines whether an injury is likely to occur from a given crash.
        - **Fatality Classification**: Predicts whether a crash is likely to involve in fatality.

    - 🌐 Deploy an interactive **Streamlit dashboard** for public use and policy exploration

    ---

    ### 🗂️ Data Source

    The dataset used in this project is sourced from the  
    [**NYPD Motor Vehicle Collisions – Crashes (NYC Open Data)**](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)  
    — a public safety database maintained by the City of New York.

    """)

elif section == "Dataset Overview":
    st.title("📂 Dataset Overview – NYC Motor Vehicle Collisions")

    st.markdown("""
    This section provides a quick look at the **raw** and **cleaned** versions of the NYC collision dataset.

    The data includes over 5 years of vehicle collision reports across New York City with ~ 0.34 million records ,  
    containing information like **date, time, borough, street names, number of injuries, fatalities, contributing factors,** and **vehicle types** involved.
    """)

    # 🔹 Raw Preview
    st.markdown("### 🔹 Raw Dataset Preview")
    st.dataframe(df_raw.head(5))
    st.markdown(f"""
    - **Shape:** 345643 × {df_raw.shape[1]} columns  
    - Raw dataset contains missing values and unstandardized features.
    """)

    st.markdown("---")

    # ✅ Cleaned Preview
    st.markdown("### ✅ Cleaned Dataset Preview")
    st.dataframe(df_cleaned.head(5))
    st.markdown(f"""
    - **Shape after cleaning:** {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns  
    """)

    # 🧹 Cleaning Summary
    st.markdown("""
    ### 🧹 Cleaning Summary
    - Removed high-null columns (vehicle_type_4/5, contributing_factor_4/5, off_street_name)
    - Cleaned text fields (`borough`, `street names`)
    - Converted date/time columns to proper formats
    - Dropped redundant `location` column (latitude + longitude already included)
    - Standardized and encoded categorical variables
    """)

    # 📊 Column-Wise Insights
    st.markdown("### 📊 Column-Wise Insights")

    st.markdown("""
    | Column Name                  | Non-Null Count | What It Means                                                                 |
    |-----------------------------|----------------|--------------------------------------------------------------------------------|
    | crash_date                  | ✅ Full        | Date of the crash                                                              |
    | crash_time                  | ✅ Full        | Time of the crash (HH:MM format)                                              |
    | borough                     | ✅ Full           | NYC borough (e.g., BROOKLYN, MANHATTAN)                                       |
    | zip_code                    | ~67%           | ZIP code of crash location                                                    |
    | latitude, longitude         | ~92%           | Coordinates of the crash                                                      |
    | on_street_name              | ~73%           | Name of street where crash occurred                                           |
    | number_of_persons_injured  | ✅ Full        | Total number of people injured in the crash                                   |
    | number_of_persons_killed   | ✅ Full        | Total number of people killed in the crash                                    |
    | number_of_pedestrians_injured | ✅ Full     | Pedestrians injured                                                            |
    | number_of_cyclist_injured  | ✅ Full        | Cyclists injured                                                               |
    | number_of_motorist_injured| ✅ Full        | Motorists injured                                                              |
    | contributing_factor_vehicle_1 | ~99%        | Primary cause of the crash (e.g., Driver Inattention)                         |
    | vehicle_type_code1          | ~99%           | Type of vehicle involved (e.g., Sedan, Taxi)                                  |
    | year, crash_hour, day_of_week | ✅ Derived  | Engineered time-based features                                                |
    """)

    st.markdown("> ✅ This cleaned dataset is ready for visualization, modeling, and interactive prediction.")

elif section == "EDA - Part 1: Temporal & Categorical":
    st.title("📊 EDA - Part 1: Temporal & Categorical Insights")

    # ✅ Top KPIs
    st.markdown("### 🔢 Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric("🚗 Total Collisions", f"{len(df_cleaned):,}")
    kpi2.metric("📅 Year Range", f"{df_cleaned['year'].min()} - {df_cleaned['year'].max()}")
    kpi3.metric("🏙️ Top Borough", df_cleaned['borough'].value_counts().idxmax())
    
    if 'crash_hour' not in df_cleaned.columns:
        df_cleaned['crash_hour'] = pd.to_datetime(df_cleaned['crash_time'], errors='coerce').dt.hour
    kpi4.metric("💥 Peak Crash Hour", int(df_cleaned['crash_hour'].mode()[0]))


    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📍 Borough-wise Count", "📅 Yearly Trend", "⏰ Hour of Day",
        "📆 Day of Week", "📊 Injury Histogram"
    ])

    # TAB 1 — Borough-wise Count
    with tab1:
        st.markdown("### 📍 Collision Count by Borough")
        st.markdown("**Why this plot? :** " \
        "To identify which NYC boroughs experience the most traffic collisions. This helps prioritize city planning and traffic enforcement in high-risk zones.")
        
        st.image("./reports/image_12.png", caption="Collision Counts by Borough", use_container_width=True)


        st.markdown("""
        **🔍 Key Observations:**  
        - **Brooklyn** shows the highest collision count.  
        - **Staten Island** has the lowest — likely due to smaller population and less traffic density.""")

    # TAB 2 — Yearly Trend
    with tab2:
        st.markdown("### 📅 Year-wise Collision Trend")
        st.markdown("**Why this plot? :** " \
        "Reveals whether collision incidents are increasing or decreasing over the years.")

        st.image("./reports/image_5.png", caption="Collisions by Year", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - Sharp drop in 2020 possibly due to COVID-19 lockdowns.  
        - Gradual recovery seen post-2020.
        """)

    # TAB 3 — Hour of Day
    with tab3:
        st.markdown("### ⏰ Collisions by Hour of Day")
        st.markdown("**Why this plot? :** " \
        "Identifies peak hours for collisions, linked to traffic and behavioral patterns.")

        st.image("./reports/image_11.png", caption="Collisions by Hour of Day", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - Peak hours are 8 AM and 5 PM, likely due to work commutes.  
        - Early morning hours show the least collisions.
        """)

    # TAB 4 — Day of Week
    with tab4:
        st.markdown("### 📆 Collisions by Day of the Week")
        st.markdown("**Why this plot? :** " \
        "Analyzes weekly patterns to uncover high-risk days.")
        
        st.image("./reports/image_10.png", caption="Collisions by Day of the Week", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - Friday records the highest number of accidents.  
        - Weekend collisions are slightly lower but still prominent.
        """)

    # TAB 5 — Injury Histogram
    with tab5:
        st.markdown("### 📊 Histogram of Number of Persons Injured")
        st.markdown("**Why this plot? :** " \
        "Shows the spread of injuries per incident — most are low severity.")

        st.image("./reports/image_4.png", caption="Histogram of Number of Persons Injured", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - Majority of cases involve 0 to 2 injuries.  
        - High injury events are rare but significant.
        """)

 
elif section == "EDA - Part 2: Severity & Contributing Factors":
    st.title("📊 EDA - Part 2: Severity, Vehicles, Factors & More")

    # ✅ KPI Summary
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🚑 Total Injuries", f"{df_cleaned['number_of_persons_injured'].sum():,.0f}")
    k2.metric("💀 Total Fatalities", f"{df_cleaned['number_of_persons_killed'].sum():,.0f}")
    k3.metric("🚘 Top Vehicle Type", df_cleaned['vehicle_type_code1'].mode()[0])

    # ⚠️ Get second most frequent contributing factor (excluding 'Unspecified')
    factor_counts = df_cleaned['contributing_factor_vehicle_1'].value_counts()
    factor_counts = factor_counts[factor_counts.index.str.lower() != "unspecified"]
    second_top_factor = factor_counts.index[0] if len(factor_counts) > 0 else "N/A"
    k4.metric("⚠️ Top Contributing Factor", second_top_factor)


    # Tabs for EDA Part 2
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🧠 Severity by Borough (Pie)", 
        "📊 Contributing Factors", 
        "🚗 Vehicle Types", 
        "📈 Injuries vs Fatalities (Scatter)", 
        "📉 Motorist Injured vs Total (lmplot)",
        "📦 Borough Injury Boxplot",
        "🔥 Correlation Heatmap"
    ])

    # --- TAB 1 ---
    with tab1:
        st.markdown("### 🧠 Severity Proportion by Borough")
        st.markdown("**Why this plot? :** " \
        "Shows borough-wise distribution of all collisions, helping detect geographic risk concentration.")

        st.image("./reports/image_9.png", caption="Proportion of Collisions by Borough", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - **Brooklyn** and **Queens** dominate collision volume, consistent with their size and traffic load.
        - **Staten Island** accounts for a very small percentage of crashes.
        """)

    # --- TAB 2 ---
    with tab2:
        st.markdown("### ⚠️ Top Contributing Factors")
        st.markdown("**Why this plot?** Highlights key reasons behind crashes — useful for prevention strategies.")

        st.image("./reports/image_8.png", caption="Top 10 Contributing Factors", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - **Driver Inattention/Distraction** is the leading cause.  
        - **Failure to Yield** and **Following Too Closely** are also significant contributors.
        """)

    # --- TAB 3 ---
    with tab3:
        st.markdown("### 🚗 Vehicle Types Involved")
        st.markdown("**Why this plot?** Identifies which types of vehicles are most often involved in collisions.")

        st.image("./reports/image_7.png", caption="Top 10 Vehicle Types in Collisions", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - **Sedans** are involved in the majority of incidents.  
        - **Taxis**, **SUVs**, and **Pickups** also appear frequently.
        """)

    # --- TAB 4 ---
    with tab4:
        st.markdown("### 📈 Injuries vs Fatalities by Borough")
        st.markdown("**Why this plot?** Visualizes relationship between injuries and fatalities at borough level.")

        st.image("./reports/image_3.png", caption="Injuries vs Fatalities by Borough", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - More injuries generally correlate with more fatalities.  
        - **Brooklyn** and **Bronx** show higher injury counts.
        """)

    # --- TAB 5 ---
    with tab5:
        st.markdown("### 📉 Motorist Injured vs Total Injured (Trend)")
        st.markdown("**Why this plot?** Reveals how motorist injury counts relate to total injuries in crashes.")

        st.image("./reports/image_6.png", caption=" Motorist Injured vs Total Injured", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - Strong positive linear relationship — as **motorist injuries** increase, **total injuries** do too.  
        - Few outliers exist at high injury levels.
        """)

    # --- TAB 6 ---
    with tab6:
        st.markdown("### 📦 Boxplot: Injury Distribution by Borough")
        st.markdown("**Why this plot?** Highlights boroughs with wider injury distribution and potential outliers.")

        st.image("./reports/image_2.png", caption="Injury Distribution per Collision by Borough", use_container_width=True)

        st.markdown("""
        **🔍 Key Observations:**  
        - The boxplot reveals that the median number of persons injured per collision is 0.0 across all boroughs. This means over 50% of crashes result in no injuries.
        - The interquartile range (IQR) is between 0 and 1, suggesting low severity for most incidents. However, the presence of multiple outliers indicates that some collisions still result in high injury counts, highlighting pockets of risk.""")
    with tab7:
        st.markdown("### 🔥 Correlation Heatmap")
        st.markdown("""**Why this plot?**  
        A correlation heatmap visualizes the linear relationships among numerical features in the dataset.  
        It's essential for identifying multicollinearity, spotting strong feature relationships, and informing feature selection for predictive modeling.""")

        st.image("./reports/image_1.png", caption="Correlation Heatmap of Numerical Features", use_container_width=True)

        st.markdown("""
         **🔍 Key Observations:**  
        - There is a strong correlation between `number_of_persons_injured` and `number_of_motorist_injured`, suggesting motorist injuries drive total injuries.  
        - Most other variables show low to moderate correlation, indicating relatively independent relationships.  
        - Understanding these correlations helps improve model interpretability and guides effective feature engineering.""")


elif section == "Geospatial Visualization":
    st.title("🗺️ Geospatial Visualization of NYC Collisions")

    st.markdown("""
    **Why this section?**  
    Visualizing the geographic distribution of collisions allows us to detect spatial hotspots, 
    analyze borough-wise patterns, and support decision-making for urban traffic planning and safety improvements.
    """)

    st.markdown("🔍 *Tip:* Try zooming in and out on the map for finer analysis.")

    # Filter rows with valid lat/lon
    df_geo = df_cleaned.dropna(subset=["latitude", "longitude"])

    # ✅ Random sample to improve performance
    if len(df_geo) > 50000:
        df_geo = df_geo.sample(n=50000, random_state=42)

    # Optional borough filter
    boroughs = sorted(df_geo["borough"].dropna().unique())
    selected_borough = st.selectbox("🏙️ Select Borough (Optional)", options=["All"] + list(boroughs))

    if selected_borough != "All":
        df_geo = df_geo[df_geo["borough"] == selected_borough]

    # 📍 Scatter Plot Map
    st.subheader("📍 Collision Locations Map")
    fig = px.scatter_mapbox(
        df_geo,
        lat="latitude",
        lon="longitude",
        hover_data=["crash_date", "borough", "number_of_persons_injured", "number_of_persons_killed"],
        zoom=10,
        height=600,
        color_discrete_sequence=["red"]
    )
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **🔍 Key Observations:**  
    - Dense clusters in **Manhattan**, **Brooklyn**, and high-traffic intersections.  
    - Visual clustering can support hotspot detection and resource planning.
    """)


elif section == "Predictions":
    st.title("🔮 Collision Outcome Predictions")
    st.markdown("Use trained ML models to predict injury severity and fatality likelihood based on crash context.")

    # Load Models
    reg_model = joblib.load("./models/gbr_injury_count.pkl")
    clf_injury = joblib.load("./models/classifier_is_injury.pkl")
    clf_fatal = joblib.load("./models/classifier_is_fatal.pkl")

    # ✅ Clean Vehicle Type Inputs
    def clean_vehicle_list(series, min_len=3):
        allowed_exceptions = {"ambulance", "all terrain"}

        # Normalize, remove NaN/empty/short strings/numeric-only
        raw_items = [
            s.strip() for s in series.dropna().unique()
            if isinstance(s, str) and s.strip().lower() not in {"unspecified", "unknown", "nan"} 
            and len(s.strip()) >= min_len and not s.strip().isdigit()
        ]

        # Remove duplicates case-insensitively
        unique_lower = {}
        for item in raw_items:
            key = item.lower()
            if key not in unique_lower:
                unique_lower[key] = item

        # Filter out unwanted A/a-starting types except exceptions
        cleaned = [
            val for key, val in unique_lower.items()
            if not (key.startswith('a') and key not in allowed_exceptions)
        ]

        return sorted(cleaned)


    vehicles_clean = clean_vehicle_list(df_cleaned["vehicle_type_code1"])
    factors_clean = clean_vehicle_list(df_cleaned["contributing_factor_vehicle_1"])

    boroughs = sorted(df_cleaned["borough"].dropna().unique())
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    years = list(range(2025, 2019, -1))

    def encode_input(b, f, v, h, d, y):
        enc = lambda val, vals: pd.Series(val).astype("category").cat.set_categories(sorted(vals)).cat.codes[0]
        return pd.DataFrame([[enc(b, boroughs),
                              enc(f, factors_clean),
                              enc(v, vehicles_clean),
                              h,
                              weekdays.index(d),
                              y]],
                            columns=["borough_enc", "factor_enc", "vehicle_enc", "crash_hour", "day_enc", "year"])

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["🔢 Injury Count", "🚑 Injury Prediction", "⚰️ Fatality Prediction"])

    # ➤ TAB 1: Injury Count (Regression)
    with tab1:
        st.header("🔢 Predict Number of Persons Injured")
        b = st.selectbox("🏙️ Borough", boroughs, key="inj_cnt_b")
        f = st.selectbox("⚠️ Contributing Factor", factors_clean, key="inj_cnt_f")
        v = st.selectbox("🚘 Vehicle Type", vehicles_clean, key="inj_cnt_v")
        h = st.slider("⏰ Crash Hour", 0, 23, 12, key="inj_cnt_h")
        d = st.selectbox("📆 Day of Week", weekdays, key="inj_cnt_d")
        y = st.selectbox("📅 Year", years, key="inj_cnt_y")

        X_pred = encode_input(b, f, v, h, d, y)
        if st.button("🔍 Predict Injuries", key="btn_cnt"):
            y_pred_log = reg_model.predict(X_pred)[0]
            injuries = int(np.round(np.expm1(y_pred_log)))
            st.success(f"🩺 Estimated Injuries: **{injuries}**")

    # ➤ TAB 2: Injury Classification
    with tab2:
        st.header("🚑 Predict Whether Injury Occurred")
        b = st.selectbox("🏙️ Borough", boroughs, key="inj_clf_b")
        f = st.selectbox("⚠️ Contributing Factor", factors_clean, key="inj_clf_f")
        v = st.selectbox("🚘 Vehicle Type", vehicles_clean, key="inj_clf_v")
        h = st.slider("⏰ Crash Hour", 0, 23, 12, key="inj_clf_h")
        d = st.selectbox("📆 Day of Week", weekdays, key="inj_clf_d")
        y = st.selectbox("📅 Year", years, key="inj_clf_y")

        X_pred = encode_input(b, f, v, h, d, y)
        if st.button("🔍 Predict Injury", key="btn_injury"):
            prob = clf_injury.predict_proba(X_pred)[0][1]
            result = clf_injury.predict(X_pred)[0]
            st.info(f"Probability of Injury: **{prob:.2%}**")
            st.success("✅ Injury Likely" if result == 1 else "🛡️ No Injury Predicted")

    # ➤ TAB 3: Fatality Classification (is_fatal)
    with tab3:
        st.header("⚰️ Predict Whether Fatality Occurred")

        b = st.selectbox("🏙️ Borough", boroughs, key="fatal_b")
        f = st.selectbox("⚠️ Contributing Factor", factors_clean, key="fatal_f")
        v = st.selectbox("🚘 Vehicle Type", vehicles_clean, key="fatal_v")
        h = st.slider("⏰ Crash Hour", 0, 23, 12, key="fatal_h")
        d = st.selectbox("📆 Day of Week", weekdays, key="fatal_d")
        y = st.selectbox("📅 Year", years, key="fatal_y")
        m = st.selectbox("🗓️ Month", list(range(1, 13)), key="fatal_m")

        # Derived features
        is_weekend = int(d in ["Saturday", "Sunday"])
        is_peak_hour = int(h in range(7, 10) or h in range(16, 19))

        # Encode input
        enc = lambda val, vals: pd.Series(val).astype("category").cat.set_categories(sorted(vals)).cat.codes[0]
        input_data = pd.DataFrame([[
            enc(b, boroughs),
            enc(f, factors_clean),
            enc(v, vehicles_clean),
            h,
            weekdays.index(d),
            y,
            m,
            is_weekend,
            is_peak_hour
        ]], columns=[
            "borough_enc", "contributing_factor_vehicle_1_enc", "vehicle_type_code1_enc",
            "crash_hour", "day_of_week_enc", "year", "month", "is_weekend", "is_peak_hour"
        ])

        if st.button("🔍 Predict Fatality", key="btn_fatal"):
            prob = clf_fatal.predict_proba(input_data)[0][1]
            result = clf_fatal.predict(input_data)[0]
            st.info(f"Probability of Fatality: **{prob:.2%}**")
            st.warning("⚠️ Fatality Likely" if result == 1 else "✅ No Fatality Predicted")





    






