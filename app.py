try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    import time
except Exception as e:
    st.error(f"Error loading dependencies: {str(e)}")
    st.info("Please check if all required packages are installed correctly.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #2e7d32;
    }
    h2 {
        color: #1976d2;
    }
    h3 {
        color: #303f9f;
    }
    </style>
""", unsafe_allow_html=True)

# Caching for performance
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def create_visualizations(df):
    figs = {}
    
    # Correlation Heatmap
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), 
                cmap="coolwarm", 
                annot=True, 
                fmt='.2f', 
                linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    figs['correlation'] = fig1

    # Class Distribution
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='type', order=df['type'].value_counts().index)
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    figs['distribution'] = fig2

    return figs

# Main menu
menu = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "📊 Upload & Train", "🔮 Prediction"],
    key="navigation"
)

# Home page
if menu == "🏠 Home":
    st.title("🤖 Machine Learning Classification Platform")
    st.markdown("""
    ### Welcome to the ML Classification Platform!
    
    This application helps you:
    1. **Upload** your dataset
    2. **Analyze** data characteristics
    3. **Train** multiple classification models
    4. **Compare** model performances
    5. **Make** predictions
    
    #### How to use:
    1. Navigate to "📊 Upload & Train" to start with your dataset
    2. Upload a CSV file containing your features and 'type' column
    3. Review the visualizations and model performances
    4. Go to "🔮 Prediction" to make predictions using the best model
    
    #### Supported Models:
    - Logistic Regression
    - Naïve Bayes
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Random Forest
    - Gradient Boosting
    """)

elif menu == "📊 Upload & Train":
    st.title("📊 Data Analysis & Model Training")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner('Loading and processing your data...'):
            df = load_data(uploaded_file)
            
            # Data Overview
            col1, col2 = st.columns(2)
            with col1:
                st.write("### 📋 Dataset Overview")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write("**Sample Data:**")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.write("### 📊 Data Statistics")
                st.write("**Numerical Features Summary:**")
                st.dataframe(df.describe(), use_container_width=True)
            
            # Missing Values Analysis
            st.write("### 🔍 Missing Values Analysis")
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.warning(f"Found {missing_values.sum()} missing values")
                st.write(missing_values[missing_values > 0])
            else:
                st.success("No missing values found in the dataset!")

            # Data Visualization
            st.write("### 📈 Data Visualization")
            figs = create_visualizations(df)
            
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.write("#### Correlation Heatmap")
                st.pyplot(figs['correlation'])
            
            with viz_col2:
                st.write("#### Class Distribution")
                st.pyplot(figs['distribution'])

            # Model Training
            st.write("### 🤖 Model Training")
            with st.spinner('Training models... Please wait...'):
                # Data Preprocessing
                df = df.dropna(axis=1, thresh=0.9 * len(df))
                df.fillna(df.median(numeric_only=True), inplace=True)

                label_encoder = LabelEncoder()
                df["type"] = label_encoder.fit_transform(df["type"])
                
                X = df.drop(columns=["type"])
                y = df["type"]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Naïve Bayes": GaussianNB(),
                    "SVM": SVC(probability=True),
                    "KNN": KNeighborsClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier()
                }

                results = {}
                best_model = None
                best_accuracy = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (model_name, model) in enumerate(models.items()):
                    status_text.text(f"Training {model_name}...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    results[model_name] = {
                        "Accuracy": accuracy,
                        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    }
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model
                    
                    progress_bar.progress((idx + 1) / len(models))
                
                status_text.text("Training completed!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()

                # Display Results
                st.write("### 📊 Model Performance Comparison")
                results_df = pd.DataFrame(results).T
                
                # Style the dataframe
                styled_results = results_df.style.format("{:.4f}")\
                    .background_gradient(cmap='YlOrRd')\
                    .highlight_max(axis=0, color='lightgreen')
                
                st.dataframe(styled_results, use_container_width=True)
                
                # Save best model
                if best_model is not None:
                    model_info = {
                        'model': best_model,
                        'scaler': scaler,
                        'label_encoder': label_encoder,
                        'feature_names': list(X.columns)
                    }
                    with open("best_model.pkl", "wb") as model_file:
                        pickle.dump(model_info, model_file)
                    st.success(f"Best model saved successfully! (Best accuracy: {best_accuracy:.4f})")

elif menu == "🔮 Prediction":
    st.title("🔮 Make Predictions")
    
    try:
        with open("best_model.pkl", "rb") as model_file:
            model_info = pickle.load(model_file)
            model = model_info['model']
            scaler = model_info['scaler']
            label_encoder = model_info['label_encoder']
            feature_names = model_info['feature_names']
        
        st.info("💡 Enter the feature values for prediction below:")
        
        # Create a more organized input interface
        col1, col2 = st.columns(2)
        input_data = {}
        
        for idx, feature in enumerate(feature_names):
            if idx % 2 == 0:
                with col1:
                    input_data[feature] = st.number_input(
                        f"📊 {feature}",
                        value=0.0,
                        help=f"Enter value for {feature}"
                    )
            else:
                with col2:
                    input_data[feature] = st.number_input(
                        f"📊 {feature}",
                        value=0.0,
                        help=f"Enter value for {feature}"
                    )
        
        if st.button("🔮 Predict", key="predict_button"):
            with st.spinner("Making prediction..."):
                input_array = np.array(list(input_data.values())).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                
                # Get prediction probabilities
                try:
                    probabilities = model.predict_proba(input_scaled)[0]
                    prob_df = pd.DataFrame({
                        'Class': label_encoder.inverse_transform(range(len(probabilities))),
                        'Probability': probabilities
                    })
                    
                    st.success(f"🎯 Predicted Class: **{predicted_label}**")
                    
                    # Display probabilities as a bar chart
                    st.write("### Prediction Probabilities")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=prob_df, x='Class', y='Probability')
                    plt.title("Prediction Probabilities by Class")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                except:
                    st.success(f"🎯 Predicted Class: **{predicted_label}**")
                    st.info("Note: Probability distribution not available for this model type.")
                
    except FileNotFoundError:
        st.error("⚠️ No trained model found. Please upload a dataset and train the model first!")
        st.info("👈 Go to the 'Upload & Train' section to train a model.")