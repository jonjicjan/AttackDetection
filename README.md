# 🤖 Machine Learning Classification Platform

A powerful and user-friendly web application for data analysis, model training, and classification predictions built with Streamlit. This platform allows users to upload their datasets, perform exploratory data analysis, train multiple machine learning models, and make predictions using the best-performing model.

## ✨ Features

### 📊 Data Analysis
- Upload and analyze CSV datasets
- Automatic data preprocessing
- Interactive data visualization
  - Correlation heatmaps
  - Class distribution plots
- Missing value analysis
- Basic statistics and data overview

### 🎯 Model Training
- Support for multiple classification algorithms:
  - Logistic Regression
  - Naïve Bayes
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Gradient Boosting
- Automatic model comparison
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Progress tracking during training
- Best model selection and persistence

### 🔮 Prediction
- Easy-to-use prediction interface
- Real-time predictions
- Probability distribution visualization
- Support for multiple features

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

Run the application using Streamlit:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📖 Usage Guide

### 1. Data Upload
- Navigate to "Upload & Train" section
- Upload your CSV file
- The file should contain:
  - Feature columns (numerical)
  - A 'type' column for classification labels

### 2. Model Training
- After uploading data, the application will:
  - Show data statistics and visualizations
  - Train multiple models
  - Display performance comparisons
  - Save the best performing model

### 3. Making Predictions
- Go to the "Prediction" section
- Enter feature values
- Click "Predict" to get classification results
- View prediction probabilities (if available)

## 🔧 Technical Details

### Data Preprocessing
- Automatic handling of missing values
- Feature scaling using StandardScaler
- Label encoding for target variable

### Model Selection
- Trains multiple classification models
- Automatically selects the best performing model
- Uses cross-validation for reliable performance estimation

### Performance Metrics
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1 Score (weighted average)

## 📝 Requirements

See `requirements.txt` for a complete list of dependencies:
- streamlit==1.43.0
- pandas==2.1.0
- numpy==1.24.3
- scikit-learn==1.3.0
- seaborn==0.12.2
- matplotlib==3.7.2

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/your-repo-name/issues).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- scikit-learn team for machine learning tools
- The open-source community

## 📞 Support

For support, email your@email.com or create an issue in the repository. 
