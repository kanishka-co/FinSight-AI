import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
from scipy import sparse

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent)
MODEL_PATH = os.path.join(BASE_PATH, "finsight_churn_model.joblib")
DATA_PATH = os.path.join(BASE_PATH, "telco_churn_cleaned.csv")

st.set_page_config(page_title="FinSight AI", page_icon="📊", layout="wide")

st.markdown('\n<style>\n/* FinSight Premium UI Theme */\n/* App background and typography */\n.stApp {\n    background: linear-gradient(145deg, #f5f7fb 0%, #eef2f8 100%);\n    color: #17243b;\n}\n.block-container {\n    padding-top: 2rem;\n    padding-bottom: 3rem;\n    max-width: 1500px;\n}\nh1, h2, h3 {\n    color: #142541 !important;\n    letter-spacing: -0.025em;\n}\nh1 { font-weight: 750 !important; }\nh2, h3 { font-weight: 650 !important; }\n\n/* Sidebar */\nsection[data-testid="stSidebar"] {\n    background: linear-gradient(180deg, #101d35 0%, #192c4b 100%);\n    border-right: 1px solid rgba(255,255,255,0.08);\n}\nsection[data-testid="stSidebar"] * {\n    color: #edf3ff;\n}\nsection[data-testid="stSidebar"] [data-testid="stRadio"] label {\n    padding: 0.45rem 0.55rem;\n    border-radius: 8px;\n}\nsection[data-testid="stSidebar"] hr {\n    border-color: rgba(255,255,255,0.16);\n}\n\n/* Metrics */\ndiv[data-testid="stMetric"] {\n    background: #ffffff;\n    border: 1px solid #e3eaf4;\n    border-radius: 14px;\n    padding: 18px 20px;\n    box-shadow: 0 4px 16px rgba(24, 45, 78, 0.045);\n}\ndiv[data-testid="stMetricLabel"] {\n    color: #61718a;\n    font-size: 0.88rem;\n}\ndiv[data-testid="stMetricValue"] {\n    color: #173a67;\n    font-weight: 750;\n}\n\n/* Containers and charts */\ndiv[data-testid="stVerticalBlockBorderWrapper"] {\n    border-radius: 14px;\n}\ndiv[data-testid="stPlotlyChart"],\ndiv[data-testid="stImage"],\ndiv[data-testid="stTable"] {\n    border-radius: 12px;\n}\ndiv[data-testid="stAlert"] {\n    border-radius: 12px;\n}\n\n/* Inputs */\ndiv[data-baseweb="input"] > div,\ndiv[data-baseweb="select"] > div,\ndiv[data-baseweb="textarea"] > div {\n    border-radius: 10px;\n}\n.stTextInput input, .stNumberInput input, .stTextArea textarea {\n    background: #ffffff;\n}\n\n/* Buttons */\n.stButton > button,\n.stDownloadButton > button,\ndiv[data-testid="stFormSubmitButton"] > button {\n    border-radius: 10px;\n    border: 1px solid #244b7b;\n    font-weight: 650;\n    transition: all 0.15s ease;\n}\n.stButton > button[kind="primary"],\ndiv[data-testid="stFormSubmitButton"] > button[kind="primary"] {\n    background: #193d68;\n    color: white;\n}\n.stButton > button:hover,\n.stDownloadButton > button:hover {\n    border-color: #477cb6;\n    box-shadow: 0 4px 12px rgba(25, 61, 104, 0.12);\n}\n\n/* Tabs and dividers */\nbutton[data-baseweb="tab"] {\n    font-weight: 600;\n}\nhr {\n    border-color: #dfe6f0;\n}\n\n/* File uploader */\n[data-testid="stFileUploader"] {\n    background: rgba(255,255,255,0.78);\n    border: 1px dashed #bdcce0;\n    border-radius: 14px;\n    padding: 10px;\n}\n\n/* Make narrow screens comfortable */\n@media (max-width: 768px) {\n    .block-container {\n        padding: 1rem 1rem 2rem;\n    }\n    div[data-testid="stMetric"] {\n        padding: 12px;\n    }\n}\n</style>\n', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    return data

try:
    artifact = load_model()
    model = artifact["model"]
    threshold = artifact["threshold"]
    df = load_data()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()

FEATURE_COLUMNS = [c for c in df.columns if c not in ["customerID", "Churn"]]

@st.cache_resource
def load_shap_explainer():
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    background_raw = df[FEATURE_COLUMNS].sample(n=min(100, len(df)), random_state=42)
    background = preprocessor.transform(background_raw)
    if sparse.issparse(background):
        background = background.toarray()
    names = preprocessor.get_feature_names_out()
    background = pd.DataFrame(background, columns=names)
    return shap.LinearExplainer(classifier, background), names

@st.cache_data
def get_global_shap_values():
    preprocessor = model.named_steps["preprocessor"]
    sample = df[FEATURE_COLUMNS].sample(n=min(200, len(df)), random_state=7)
    transformed = preprocessor.transform(sample)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    explainer, names = load_shap_explainer()
    return explainer(pd.DataFrame(transformed, columns=names))

st.sidebar.title("FinSight AI")
st.sidebar.caption("Customer Intelligence Platform")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Churn Analytics",
        "Predict Customer",
        "Explainability",
        "Financial Document Intelligence",
    ],
)
st.sidebar.divider()
st.sidebar.caption("ML model: Logistic Regression")
st.sidebar.caption(f"Decision threshold: {threshold:.0%}")

if page == "Overview":
    st.title("📊 Customer Analytics Overview")
    st.write("Explore customer retention patterns from the IBM Telco Customer Churn dataset.")
    total_customers = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    retained = int((df["Churn"] == "No").sum())
    churn_rate = churned / total_customers * 100 if total_customers else 0
    monthly_revenue = df["MonthlyCharges"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Customer Churn Rate", f"{churn_rate:.1f}%")
    c4.metric("Monthly Charges", f"${monthly_revenue:,.0f}")
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Customer Status")
        status_counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
        status_counts.index = ["Retained", "Churned"]
        st.bar_chart(status_counts)
    with right:
        st.subheader("Customers by Contract")
        st.bar_chart(df["Contract"].value_counts())
    st.subheader("Customer Tenure Distribution")
    st.bar_chart(df.groupby("tenure").size().rename("Customers"))
    st.caption("Monthly Charges is the sum of listed customer monthly charges, not company profit or total financial revenue.")

elif page == "Churn Analytics":
    st.title("📈 Churn Analytics")
    st.write("Compare observed churn rates across different customer characteristics.")
    category = st.selectbox("Analyze churn by", ["Contract", "InternetService", "PaymentMethod", "TechSupport", "PaperlessBilling", "OnlineSecurity"])
    churn_rate_by_category = (
        pd.crosstab(df[category], df["Churn"], normalize="index")
        .reindex(columns=["No", "Yes"], fill_value=0)["Yes"].mul(100)
        .sort_values(ascending=False).rename("Churn Rate (%)")
    )
    st.subheader(f"Churn Rate by {category}")
    st.bar_chart(churn_rate_by_category)
    st.dataframe(churn_rate_by_category.reset_index(), use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Monthly Charges by Customer Status")
    charges = df.groupby("Churn")["MonthlyCharges"].mean().rename(index={"No": "Retained", "Yes": "Churned"})
    st.bar_chart(charges)
    st.subheader("Tenure by Customer Status")
    tenure = df.groupby("Churn")["tenure"].mean().rename(index={"No": "Retained", "Yes": "Churned"})
    st.bar_chart(tenure)
    st.caption("These are observed associations in this dataset. They do not establish that a characteristic causes churn.")

elif page == "Predict Customer":
    st.title("🔍 Customer Churn Prediction")
    st.write("Enter a customer's details to estimate their churn probability.")
    st.info("Demonstration model trained on the IBM Telco Customer Churn dataset.")
    with st.form("customer_form"):
        st.subheader("Customer Details")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        col3, col4 = st.columns(2)
        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        with col4:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
        submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)
    if submitted:
        if tenure == 0 and total_charges > 0:
            st.warning("Check the values: a zero-month tenure with positive total charges may be inconsistent.")
        customer = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
        }])
        probability = model.predict_proba(customer)[0, 1]
        prediction = int(probability >= threshold)
        st.session_state["last_customer"] = customer
        st.divider()
        st.subheader("Prediction Results")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Estimated Churn Probability", f"{probability:.1%}")
            st.progress(float(probability))
        with c2:
            if prediction:
                st.error("Higher churn risk")
            else:
                st.success("Lower churn risk")
            st.caption(f"Decision threshold: {threshold:.0%}")
        if prediction:
            st.write("The estimated probability meets or exceeds the selected threshold. Consider reviewing this customer's circumstances.")
        else:
            st.write("The estimated probability is below the selected threshold. This does not guarantee that the customer will stay.")
        st.caption("Model probabilities may not be calibrated for real-world use. Validate before making business decisions.")

elif page == "Explainability":
    st.title("🧠 Model Explainability")
    st.write("SHAP helps explain how features contribute to this model's predictions. These explanations do not prove causation.")
    st.info("SHAP values for this Logistic Regression model are in log-odds, not percentage points of churn probability.")
    try:
        with st.spinner("Calculating SHAP explanations..."):
            global_values = get_global_shap_values()
        st.subheader("1. Overall Feature Importance")
        st.write("Features at the top have the largest average absolute contribution across the sample.")
        fig, ax = plt.subplots()
        shap.plots.bar(global_values, max_display=15, show=False)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.subheader("2. How Features Influence Predictions")
        st.write("Each dot represents a customer. Color represents feature value; left/right indicates the direction of the contribution.")
        fig, ax = plt.subplots()
        shap.plots.beeswarm(global_values, max_display=15, show=False)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.divider()
        st.subheader("3. Explain One Customer")
        last_customer = st.session_state.get("last_customer")
        if last_customer is not None:
            choice = st.radio("Choose a customer to explain", ["Most recent prediction", "Choose a dataset customer"], horizontal=True)
        else:
            choice = "Choose a dataset customer"
            st.caption("You can explain a customer you predicted on the Predict Customer page, or choose a dataset example below.")

        if choice == "Most recent prediction" and last_customer is not None:
            customer_to_explain = last_customer[FEATURE_COLUMNS].copy()
            label = "Your most recent prediction"
        else:
            selected_row = st.selectbox("Choose a sample customer", options=list(range(min(100, len(df)))), format_func=lambda row: f"Sample customer {row + 1}")
            customer_to_explain = df.iloc[[selected_row]][FEATURE_COLUMNS].copy()
            label = f"Sample customer {selected_row + 1}"

        probability = model.predict_proba(customer_to_explain)[0, 1]
        st.write(f"**{label}**")
        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.caption(f"Decision threshold: {threshold:.0%}. " + ("Classified as higher churn risk." if probability >= threshold else "Classified as lower churn risk."))

        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transform(customer_to_explain)
        if sparse.issparse(transformed):
            transformed = transformed.toarray()
        explainer, names = load_shap_explainer()
        transformed = pd.DataFrame(transformed, columns=names)
        individual_values = explainer(transformed)
        fig, ax = plt.subplots()
        shap.plots.waterfall(individual_values[0], max_display=15, show=False)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption("Red features push the model output higher; blue features push it lower. These are model contributions, not proof of cause.")
    except Exception as error:
        st.error(f"Could not create SHAP explanations: {error}")
        st.info("If SHAP is missing in the Streamlit environment, run: !pip install -q shap")

elif page == "Financial Document Intelligence":
    from document_intelligence import (
        load_embedding_model,
        process_pdf,
        search_document_data,
        ask_groq,
    )
    import hashlib

    st.title("📄 Financial Document Intelligence")
    st.write(
        "Upload a financial report and ask questions about "
        "revenue, profitability, and other financial metrics."
    )
    st.info(
        "Answers depend on extracted PDF text and retrieved passages. "
        "Verify important figures against the cited report pages."
    )

    uploaded_file = st.file_uploader(
        "Upload a financial PDF",
        type=["pdf"],
        key="financial_pdf_upload",
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(pdf_bytes).hexdigest()

        st.caption(f"Selected file: {uploaded_file.name}")

        saved_hash = st.session_state.get("financial_file_hash")
        if saved_hash is not None and saved_hash != current_hash:
            st.session_state.pop("financial_document", None)
            st.session_state.pop("financial_file_hash", None)
            st.session_state.pop("financial_file_name", None)

        if st.button("📄 Process PDF", type="primary",
                     key="process_financial_pdf"):
            try:
                with st.spinner(
                    "Extracting text and creating embeddings..."
                ):
                    model = load_embedding_model()
                    document = process_pdf(pdf_bytes, model)

                    st.session_state["financial_document"] = document
                    st.session_state["financial_file_hash"] = current_hash
                    st.session_state["financial_file_name"] = uploaded_file.name

                st.success("PDF processed successfully!")

            except Exception as error:
                st.error(f"Could not process PDF: {error}")

        document = st.session_state.get("financial_document")
        if (
            document is not None
            and st.session_state.get("financial_file_hash") == current_hash
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Readable pages", len(document["pages"]))
            col2.metric("Text chunks", len(document["chunks"]))
            col3.metric(
                "Embedding dimensions",
                document["embeddings"].shape[1],
            )

            st.divider()
            st.subheader("Ask your financial report")

            question = st.text_input(
                "Your question",
                placeholder="What was the operating income in 2025?",
                key="financial_document_question",
            )

            if st.button("🔎 Ask FinSight AI", key="ask_financial_question"):
                if not question.strip():
                    st.warning("Please enter a question first.")
                else:
                    try:
                        model = load_embedding_model()

                        with st.spinner("Searching the report..."):
                            results = search_document_data(
                                question, document, model, top_k=5
                            )

                        if not results:
                            st.warning(
                                "No relevant evidence was retrieved. "
                                "Try rephrasing your question."
                            )
                        else:
                            with st.spinner(
                                "Generating an evidence-based answer..."
                            ):
                                answer = ask_groq(question, results, document=document)

                            st.subheader("Answer")
                            st.markdown(answer)

                            st.subheader("Retrieved evidence")
                            for index, item in enumerate(results, start=1):
                                with st.expander(
                                    f"Evidence {index} — PDF page {item['page']}"
                                ):
                                    st.write(item["text"])

                    except Exception as error:
                        st.error(f"Unable to answer the question: {error}")

    else:
        st.caption("Upload a PDF and click Process PDF to get started.")

