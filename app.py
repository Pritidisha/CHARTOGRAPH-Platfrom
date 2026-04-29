import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import xgboost as xgb
from groq import Groq
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# CONFIG
st.set_page_config(page_title="CHARTOGRAPH", layout="wide")
st.title("CHARTOGRAPH")


# INIT GROQ
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# FILE READER
def read_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    else:
        st.error("Unsupported file format")
        return None


# DATASET QUALITY
def evaluate_quality(df):
    rows, cols = df.shape
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    score = 100
    if missing > 0:
        score -= 15
    if rows < 100:
        score -= 20
    if cols < 3:
        score -= 10
    if duplicates > 0:
        score -= 10

    return score, missing, duplicates



# AI GRAPH ANALYSIS
def analyze_graph_with_ai(summary):
    prompt = f"""
You are a senior data analyst.

Analyze this chart summary:

{summary}

Provide:
1. What it shows
2. Key patterns
3. Risks/anomalies
4. Business insights
5. Actionable recommendations

Explain simply.
Don't explicitly state personal pronouns or that you are a bot/business analyst. Be concise and insightful. 
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content



#explain_mode = st.checkbox("Enable Explanations")
# UPLOAD
file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "json"])

with st.expander("ℹ️ About Dataset Upload"):
    st.markdown("""
    **What is this step?**  
    Upload your dataset to begin analysis.

    **Why it matters:**  
    The quality and structure of your data directly impact insights, visualizations, and model accuracy.

    **Tip:** 
    Clean, well-structured datasets lead to better results.
    """)



if file:

    if "last_file" not in st.session_state or st.session_state["last_file"] != file.name:
        st.session_state.clear()
        st.session_state["last_file"] = file.name
        
    df = read_file(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # QUALITY
    st.header("Dataset Quality Evaluation")
    score, missing, duplicates = evaluate_quality(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", missing)
    c4.metric("Quality Score", f"{score}/100")

    with st.expander("ℹ️ About Dataset Quality"):
        st.markdown("""
        **What is Dataset Quality?**  
        This checks how reliable and usable your data is.

        **What affects quality?**
        - Missing values reduce accuracy  
        - Duplicate rows distort insights  
        - Small datasets limit learning  
        - Too few columns restrict analysis  

        **Goal:**  
        A higher score means your data is more suitable for analysis and machine learning.
        """)


    # DATA CLEANING OPTIONS
    st.header("Data Cleaning Options")

    df_cleaned = df.copy()

    #Missing Values Handling
    if missing > 0:
        st.write("🧩 Missing values detected in the dataset. Select how you would like to handle them.")

        missing_option = st.selectbox(
            "Choose missing value strategy",
            ["Do Nothing", "Drop Rows", "Fill with Mean/Mode", "Fill with Median/Mode"]
        )

        if missing_option != "Do Nothing":

            if st.button("Apply"):

                if missing_option == "Drop Rows":
                    df_cleaned = df_cleaned.dropna()

                elif missing_option == "Fill with Mean/Mode":
                    for col in df_cleaned.columns:
                        if df_cleaned[col].dtype in ["int64", "float64"]:
                            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                        else:
                            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

                elif missing_option == "Fill with Median/Mode":
                    for col in df_cleaned.columns:
                        if df_cleaned[col].dtype in ["int64", "float64"]:
                            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                        else:
                            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

                st.success("Missing values handled successfully!")
                df = df_cleaned  # update main dataframe


    # Duplicate Handling
    if duplicates > 0:
        st.write("🧩 Duplicate values detected in the dataset. Press the button below to handle this issue.")

        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates removed successfully!")

    with st.expander("ℹ️ About Data Cleaning"):
        st.markdown("""
        **What is Data Cleaning?**  
        Real-world data is often messy. This step helps fix common issues.

        **What can you do here?**
        - Remove missing values  
        - Fill missing values with averages or common values  
        - Remove duplicate entries  

        **Why it matters:**  
        Machine learning models and visualizations perform poorly on dirty data.
        """)

    # CHART EXPLANATIONS (UI GUIDE)
    chart_info = {
        "Scatter": {
            "desc": "Shows relationship between two numerical variables.",
            "use": "Best for detecting correlation or patterns.",
            "warn": "Both X and Y should be numeric."
        },
        "Line": {
            "desc": "Displays trends over time or ordered data.",
            "use": "Best for time-series or continuous progression.",
            "warn": "X-axis should ideally be time or ordered numeric."
        },
        "Area": {
            "desc": "Similar to line chart but emphasizes magnitude.",
            "use": "Useful for cumulative trends.",
            "warn": "Not ideal for categorical comparisons."
        },
        "Bar": {
            "desc": "Compares values across categories.",
            "use": "Best for categorical vs numeric comparison.",
            "warn": "Categorical X + Numeric Y works best."
        },
        "Grouped Bar": {
            "desc": "Compares multiple groups side-by-side.",
            "use": "Useful for subgroup comparisons.",
            "warn": "Too many categories will clutter the chart."
        },
        "Stacked Bar": {
            "desc": "Shows composition within categories.",
            "use": "Best for part-to-whole relationships.",
            "warn": "Hard to read if too many segments."
        },
        "Histogram": {
            "desc": "Shows distribution of a single variable.",
            "use": "Best for understanding spread and skewness.",
            "warn": "Works best with numeric data."
        },
        "Box": {
            "desc": "Shows distribution and outliers.",
            "use": "Best for comparing spread across categories.",
            "warn": "Needs numeric Y."
        },
        "Violin": {
            "desc": "Shows distribution shape and density.",
            "use": "More detailed than box plot.",
            "warn": "Can be confusing for beginners."
        },
        "Density Contour": {
            "desc": "Shows density regions in 2D.",
            "use": "Best for identifying clusters.",
            "warn": "Requires numeric columns."
        },
        "Density Heatmap": {
            "desc": "2D frequency heatmap.",
            "use": "Good for large datasets.",
            "warn": "Only numeric columns allowed."
        },
        "Pie": {
            "desc": "Shows proportion of categories.",
            "use": "Best for small number of categories.",
            "warn": "Avoid if too many categories."
        },
        "Donut": {
            "desc": "Pie chart with center space.",
            "use": "Same as pie but cleaner look.",
            "warn": "Same limitations as pie chart."
        },
        "Treemap": {
            "desc": "Hierarchical part-to-whole visualization.",
            "use": "Best for nested categories.",
            "warn": "Hard to read with too many levels."
        },
        "Sunburst": {
            "desc": "Radial hierarchical chart.",
            "use": "Shows multi-level relationships.",
            "warn": "Can become cluttered quickly."
        },
        "Funnel": {
            "desc": "Shows step-wise reduction.",
            "use": "Best for processes like sales funnels.",
            "warn": "Requires meaningful stage order."
        },
        "Correlation Heatmap": {
            "desc": "Shows correlation between numeric variables.",
            "use": "Best for feature relationships.",
            "warn": "Only numeric columns are used."
        },
        "Pair Plot": {
            "desc": "Matrix of scatter plots.",
            "use": "Explore all relationships at once.",
            "warn": "Slow for large datasets."
        },
        "3D Scatter": {
            "desc": "3D relationship between variables.",
            "use": "For complex multi-variable patterns.",
            "warn": "Hard to interpret sometimes."
        },
        "Bubble Chart": {
            "desc": "Scatter plot with size dimension.",
            "use": "Adds third variable via size.",
            "warn": "Too many points reduce clarity."
        }
    }

    # VISUALIZATION
    st.header("Advanced Data Visualization")

    chart_types = [
        "Scatter", "Line", "Area",
        "Bar", "Grouped Bar", "Stacked Bar",
        "Histogram", "Box", "Violin",
        "Pie", "Donut",
        "Treemap", "Sunburst",
        "Funnel",
        "Density Contour", "Density Heatmap",
        "Correlation Heatmap",
        "Pair Plot",
        "3D Scatter",
        "Bubble Chart"
    ]

    chart = st.selectbox("Select Chart Type", chart_types)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    fig = None

    # Charts that need no axes
    if chart == "Correlation Heatmap":
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True)

    elif chart == "Pair Plot":
        fig = px.scatter_matrix(df[numeric_cols])

    # Charts that need only X
    elif chart in ["Histogram"]:
        x = st.selectbox("X-axis", df.columns)
        fig = px.histogram(df, x=x)

    # Charts that need X and Y
    elif chart in ["Scatter", "Line", "Area", "Bar",
                "Grouped Bar", "Stacked Bar",
                "Box", "Violin",
                "Density Contour", "Density Heatmap"]:

        x = st.selectbox("X-axis", df.columns)
        y = st.selectbox("Y-axis", df.columns)
        
        # SMART VALIDATION WARNINGS
        if chart in ["Scatter", "Line", "Area", "Density Contour", "Density Heatmap"]:
            if x not in numeric_cols or y not in numeric_cols:
                st.warning(f"{chart} works best with numeric columns.")

        if chart == "Bar":
            if not (x in categorical_cols and y in numeric_cols):
                st.warning("Bar chart works best with categorical X and numeric Y.")

        if chart == "Histogram":
            if x not in numeric_cols:
                st.warning("Histogram should use numeric columns.")

        if chart in ["Pie", "Donut"]:
            if x not in categorical_cols:
                st.warning("Pie charts require categorical data.")

        if chart == "Scatter":
            fig = px.scatter(df, x=x, y=y)
        elif chart == "Line":
            fig = px.line(df, x=x, y=y)
        elif chart == "Area":
            fig = px.area(df, x=x, y=y)
        elif chart == "Bar":
            if x in categorical_cols and y in numeric_cols:
                df_grouped = df.groupby(x)[y].mean().reset_index()
                fig = px.bar(df_grouped, x=x, y=y)
            else:
                fig = px.bar(df, x=x, y=y)
        elif chart == "Grouped Bar":
            fig = px.bar(df, x=x, y=y, color=x, barmode="group")
        elif chart == "Stacked Bar":
            fig = px.bar(df, x=x, y=y, color=x, barmode="stack")
        elif chart == "Box":
            fig = px.box(df, x=x, y=y)
        elif chart == "Violin":
            fig = px.violin(df, x=x, y=y, box=True)
        elif chart == "Density Contour":
            fig = px.density_contour(df, x=x, y=y)
        elif chart == "Density Heatmap":
            fig = px.density_heatmap(df, x=x, y=y)

    # Hierarchical charts
    elif chart in ["Pie", "Donut", "Treemap", "Sunburst", "Funnel"]:

        x = st.selectbox("Category Column", df.columns)
        y = st.selectbox("Value Column", numeric_cols)

        if chart == "Pie":
            fig = px.pie(df, names=x, values=y)
        elif chart == "Donut":
            fig = px.pie(df, names=x, values=y, hole=0.4)
        elif chart == "Treemap":
            fig = px.treemap(df, path=[x], values=y)
        elif chart == "Sunburst":
            fig = px.sunburst(df, path=[x], values=y)
        elif chart == "Funnel":
            fig = px.funnel(df, x=y, y=x)

    # Advanced charts
    elif chart == "3D Scatter":
        x = st.selectbox("X-axis", numeric_cols)
        y = st.selectbox("Y-axis", numeric_cols)
        z = st.selectbox("Z-axis", numeric_cols)
        fig = px.scatter_3d(df, x=x, y=y, z=z)

    elif chart == "Bubble Chart":
        x = st.selectbox("X-axis", numeric_cols)
        y = st.selectbox("Y-axis", numeric_cols)
        size = st.selectbox("Size Column", numeric_cols)
        fig = px.scatter(df, x=x, y=y, size=size)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    

    def classify_columns(df):
        col_info = {}

        for col in df.columns:
            unique_vals = df[col].nunique()
            dtype = df[col].dtype

            if dtype == "object":
                if unique_vals <= 15:
                    col_info[col] = "categorical_low"
                else:
                    col_info[col] = "categorical_high"

            elif np.issubdtype(dtype, np.number):
                if unique_vals <= 10:
                    col_info[col] = "categorical_numeric"
                else:
                    col_info[col] = "continuous"

            else:
                col_info[col] = "unknown"

        return col_info


    with st.expander("ℹ️ About Chart", expanded=False):
        info = chart_info.get(chart, None)
    
        if info:
            st.markdown(f"**What it does:** {info['desc']}")
            st.markdown(f"**When to use:** {info['use']}")
            st.markdown(f"**Important:** {info['warn']}")

    def recommend_charts(df):

        recommendations = []
        col_info = classify_columns(df)

        num_cols = [c for c in df.columns if col_info[c] == "continuous"]
        cat_low = [c for c in df.columns if col_info[c] == "categorical_low"]

        # Remove ID-like columns (very high uniqueness)
        num_cols = [c for c in num_cols if df[c].nunique() < len(df) * 0.9]

        # 1. Continuous vs Continuous = Scatter
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().abs()

            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    pairs.append((c1, c2, corr.loc[c1, c2]))

            pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

            for c1, c2, val in pairs:
                if val > 0.3:  # ignore weak/noise correlations
                    recommendations.append({
                        "chart": "Scatter",
                        "x": c1,
                        "y": c2,
                        "reason": f"Moderate/strong relationship ({round(val,2)}) between {c1} and {c2}"
                    })
                    break

        # 2. Categorical vs Continuous = Bar
        if cat_low and num_cols:
            recommendations.append({
                "chart": "Bar",
                "x": cat_low[0],
                "y": num_cols[0],
                "agg": "mean",
                "reason": f"Compare average {num_cols[0]} across categories of {cat_low[0]}"
            })

        # 3. Distribution = Histogram
        if num_cols:
            # pick most variable column
            variances = df[num_cols].var().sort_values(ascending=False)
            best_col = variances.index[0]

            recommendations.append({
                "chart": "Histogram",
                "x": best_col,
                "reason": f"Understand distribution of {best_col}"
            })

        # 4. Outliers = Box Plot
        if cat_low and num_cols:
            recommendations.append({
                "chart": "Box",
                "x": cat_low[0],
                "y": num_cols[0],
                "reason": f"Detect spread and outliers of {num_cols[0]} across {cat_low[0]}"
            })

        # 5. Correlation Heatmap
        if len(num_cols) >= 3:
            recommendations.append({
                "chart": "Correlation Heatmap",
                "reason": "Overview of relationships between numerical features"
            })

        return recommendations
    


# SMART RECOMMENDATIONS UI
    st.header("Smart Chart Recommendations")

    if st.button("Generate Chart Recommendations"):
        st.session_state["recs"] = recommend_charts(df)

    if "recs" in st.session_state:

        recs = st.session_state["recs"]

        if recs:

            for i, r in enumerate(recs):

                st.success(f"📊 Recommended: {r['chart']}")
                st.write(f"Reason: {r['reason']}")

                fig = None

                #Generate Graph

                x = r.get("x")
                y = r.get("y")
                if x == y:
                    alt_cols = [col for col in df.columns if col != x]
                    if alt_cols:
                        y = alt_cols[0]  # pick a different column
                    else:
                        y = None  # fallback

                if r["chart"] == "Scatter" and x and y:
                    fig = px.scatter(df, x=x, y=y)

                elif r["chart"] == "Line" and x and y:
                    fig = px.line(df, x=x, y=y)

                elif r["chart"] == "Histogram":
                    fig = px.histogram(df, x=x)

                elif r["chart"] == "Bar" and x and y:
                    fig = px.bar(df, x=x, y=y)

                elif r["chart"] == "Box" and x and y:
                    fig = px.box(df, x=x, y=y)

                elif r["chart"] == "Density Heatmap" and x and y:
                    fig = px.density_heatmap(df, x=x, y=y)

                elif r["chart"] == "Correlation Heatmap":

                    corr = df.corr(numeric_only=True)

                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r"
                    )

                #Show Graph

                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{r['chart']}_{i}")

                #AI Graph Explanation

                try:
                    summary = f"{r['chart']} chart using columns {r.get('x')} and {r.get('y')}. Dataset rows: {len(df)}"

                    analysis = analyze_graph_with_ai(summary)

                    st.info("AI Insight")
                    st.write(analysis)

                except:
                    st.warning("AI analysis unavailable.")

                st.write("---")

        else:
            st.warning("No specific recommendation found.")


    # AUTOMATIC EDA REPORT
    st.header("Automatic EDA Report")

    if st.button("Generate EDA Report"):
        st.write("### Basic Info")
        st.write(df.describe())

        st.write("### Missing Values")
        st.write(df.isnull().sum())

        st.write("### Correlation Matrix")
        st.dataframe(df.corr(numeric_only=True))

    with st.expander("ℹ️ About EDA Report"):
            st.markdown("""
            **What is EDA (Exploratory Data Analysis)?**  
            A quick statistical overview of your dataset.

            **Includes:**
            - Mean, median, standard deviation  
            - Missing value counts  
            - Correlation between variables  

            **Why it matters:**  
            Helps you understand your data before building models.
            """)

    # MACHINE LEARNING
    st.header("Machine Learning")

    target = st.selectbox("Target Variable", df.columns)
    features = st.multiselect("Feature Variables", df.columns)

    with st.expander("ℹ️ About Model Building"):
            st.markdown("""
            **What is happening here?**
            The system builds predictive models using your data.

            **Two types of problems:**
            - Regression → Predict numbers (e.g., price)
            - Classification → Predict categories (e.g., yes/no)

            **What happens internally:**
            - Data is cleaned and transformed
            - Multiple models are trained

            **Why it matters:**
            Turns your data into a decision-making tool.
            """)

    if target and features:

        X = df[features]
        y_data = df[target]

        # Detect problem type
        if y_data.dtype == "object" or y_data.nunique() <= 10:
            problem_type = "classification"
        else:
            problem_type = "regression"

        st.success(f"Detected Problem Type: {problem_type.upper()}")


        cat = X.select_dtypes(include="object").columns
        num = X.select_dtypes(exclude="object").columns

        # Numerical pipeline = impute + scale
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical pipeline = impute + encode
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine both
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num),
            ("cat", cat_pipeline, cat),
        ])
                

        X_processed = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_data, test_size=0.2, random_state=42
        )

        if st.button("Train Models"):

            if problem_type == "regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(),
                    "Lasso Regression": Lasso(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "KNN Regressor": KNeighborsRegressor(),
                    "SVR": SVR(),
                    "XGBoost": xgb.XGBRegressor()
                }

            else:  # classification
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest Classifier": RandomForestClassifier()
                }

            results = []
            trained_models = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if problem_type == "regression":
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    results.append([name, mae, r2])

                else:
                    acc = accuracy_score(y_test, preds)
                    results.append([name, acc])


                trained_models[name] = model

            
            if problem_type == "regression":
                results_df = pd.DataFrame(results, columns=["Model", "MAE", "R2"])
            else:
                results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
                        
            st.dataframe(results_df)

            
            if problem_type == "regression":
                best_name = results_df.sort_values("R2", ascending=False).iloc[0]["Model"]
            else:
                best_name = results_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
            
            best_model = trained_models[best_name]

            st.success(f"Best Model Selected: {best_name}")

            # SAVE MODEL + PREPROCESSOR 
            st.session_state["best_model"] = best_model
            st.session_state["preprocessor"] = preprocessor
            st.session_state["features"] = features
            st.session_state["problem_type"] = problem_type

            with st.expander("ℹ️ About Model Evaluation"):
                st.markdown("""
                **What happens here?**  
                Each model is tested to see how well it performs.

                **Metrics:**
                - MAE → Prediction error  
                - R² → Accuracy of regression  
                - Accuracy → Correct classification predictions  

                **Goal:**  
                Automatically select the best-performing model.
                """)



    # PREDICTION SECTION
    if "best_model" in st.session_state:

        st.header("Prediction Tool")

        model = st.session_state["best_model"]
        preprocessor = st.session_state["preprocessor"]
        features = st.session_state["features"]
        problem_type = st.session_state["problem_type"]

        user_input = {}

        for col in features:
            if df[col].dtype == "object":
                user_input[col] = st.selectbox(col, df[col].unique())
            else:
                user_input[col] = st.number_input(col, value=float(df[col].mean()))

        if st.button("Predict Outcome"):

            input_df = pd.DataFrame([user_input])
            input_processed = preprocessor.transform(input_df)

            prediction = model.predict(input_processed)[0]

            #Regression Output
            if problem_type == "regression":
                st.success(f"Predicted Value: {round(prediction, 2)}")

            #Classification Output
            else:
                st.success(f"Predicted Class: {prediction}")

                # Show probability if available
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_processed)[0]
                    classes = model.classes_

                    prob_df = pd.DataFrame({
                        "Class": classes,
                        "Probability": probs
                    })

                    st.write("### Prediction Probabilities")
                    st.dataframe(prob_df)

            with st.expander("ℹ️ About Predictions"):
                st.markdown("""
                **What just happened?**  
                You entered new data to get a prediction from the trained model.

                **How it works:**
                - Input is processed like training data  
                - Model applies learned patterns  

                **Why it matters:**  
                Turns your model into a real-world decision tool.
                """)   
