# 💰 Loan Default Prediction System

An advanced machine learning application for predicting loan default probability using Logistic Regression and an interactive Streamlit dashboard.

## 🎯 Project Overview

This system predicts the likelihood of loan default based on borrower characteristics and loan parameters. It provides:
- **Predictive Analytics:** Real-time loan default probability prediction
- **Interactive Dashboard:** Beautiful, user-friendly Streamlit interface
- **Data Analytics:** Comprehensive exploratory data analysis
- **Model Evaluation:** Detailed performance metrics and visualizations

## 📊 Dataset

- **Total Records:** 500 loan applications
- **Features:** 11 (Principal, Terms, Age, Education, Gender, etc.)
- **Target Variable:** Loan Status (PAIDOFF, COLLECTION, COLLECTION_PAIDOFF)
- **Data Split:** 70% Training, 30% Testing

### Loan Status Distribution
- PAIDOFF: 168 loans (33.6%)
- COLLECTION: 167 loans (33.4%)
- COLLECTION_PAIDOFF: 165 loans (33.0%)

## 🤖 Machine Learning Model

- **Algorithm:** Logistic Regression
- **Purpose:** Binary Classification (Default vs. Non-Default)
- **Features Used:** 6 (Principal, Terms, Age, Gender_Male, Education_High School, Education_Masters)
- **Training Samples:** 350 (70%)
- **Test Samples:** 150 (30%)

### Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 66.67% |
| Precision | 66.67% |
| Recall | 100% |
| F1-Score | 80% |
| ROC-AUC | 0.53 |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
```bash
cd /workspaces/default_loan_prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Alternatively, install manually:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly joblib
```

## 📱 Running the Dashboard

Start the Streamlit application:
```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## 🎨 Dashboard Features

### 1. 🏠 Home Page
- Quick statistics overview
- Loan status distribution
- Key metrics and averages
- Dataset information

### 2. 🔮 Predictions
- Interactive loan parameter input
- Real-time default probability prediction
- Risk level indicators
- Probability distribution visualization
- Risk assessment guidance

### 3. 📊 Analytics
- **Demographics:** Age and gender analysis
- **Finance:** Principal and loan terms analysis
- **Trends:** Education level patterns
- **Correlations:** Feature correlation matrix

### 4. 📈 Model Performance
- Accuracy and precision metrics
- Confusion matrix visualization
- ROC curve analysis
- Detailed classification report
- Model specifications

### 5. ℹ️ About
- Project overview and objectives
- Technical stack information
- Data privacy assurance
- Quick statistics

## 📂 Project Structure

```
default_loan_prediction/
├── app.py                                      # Main Streamlit application
├── Loan_Predictive.ipynb                       # Jupyter notebook with EDA and modeling
├── logistic_regression_model.pkl               # Trained model file
├── model_features.pkl                          # Feature names list
├── Loan payments data-selected-columns.csv     # Dataset
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.28.1 |
| **Backend** | Python 3.x |
| **ML Framework** | Scikit-learn 1.3.2 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Model Serialization** | Joblib |

## 📋 Features Explained

### Input Features
- **Principal:** Loan amount ($500-$1500)
- **Terms:** Loan duration (7, 15, or 30 days)
- **Age:** Borrower age (20-60 years)
- **Gender:** Borrower gender (Male/Female)
- **Education:** Education level (High School/Bachelors/Masters)

### Output
- **Default Probability:** Percentage likelihood of loan default
- **Risk Level:** Visual indicator (Low/Medium/High)
- **Probability Distribution:** Bar chart showing both outcomes

## 📊 Data Analysis Insights

### Key Findings
1. **Terms Correlation:** Longer loan terms correlate with higher default rates
   - 7 days: ~4.8% default rate
   - 15 days: ~38.7% default rate
   - 30 days: ~43.8% default rate

2. **Gender Impact:** Male borrowers show higher default propensity
   - Female: ~60% default rate
   - Male: ~74% default rate

3. **Age Factor:** Age is a weak predictor of default
   - PAIDOFF avg age: ~39 years
   - DEFAULT avg age: ~40 years

4. **Principal:** Slightly higher principal correlates with default
   - PAIDOFF avg: ~$977
   - DEFAULT avg: ~$981

## 🎓 Usage Examples

### Example 1: Low-Risk Prediction
- Principal: $700
- Terms: 7 days
- Age: 45 years
- Gender: Female
- Education: Masters
- **Expected Result:** ~20% default probability (Low Risk ✅)

### Example 2: High-Risk Prediction
- Principal: $1400
- Terms: 30 days
- Age: 25 years
- Gender: Male
- Education: High School
- **Expected Result:** ~75% default probability (High Risk ⚠️)

## 📈 Model Interpretation

### Feature Importance (Coefficients)
1. **Gender_Male** (+0.438): Male gender increases default risk
2. **Education_High School** (-0.423): Lower education increases risk
3. **Education_Masters** (+0.166): Higher education slightly increases risk
4. **Age** (+0.037): Older age slightly increases risk
5. **Terms** (-0.006): Longer terms slightly increase risk
6. **Principal** (-0.0001): Higher principal slightly decreases risk

## 🔐 Data Privacy & Security

- ✅ All data processed locally
- ✅ No data sent to external servers
- ✅ No personal information storage
- ✅ On-demand prediction calculation
- ✅ No data sharing or third-party access

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: logistic_regression_model.pkl"
**Solution:** Ensure model files are in the correct directory
```bash
ls *.pkl  # Check for model and feature files
```

### Issue: Dashboard won't open
**Solution:** Check if port 8501 is available
```bash
streamlit run app.py --server.port 8502  # Use different port
```

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Plotly Documentation](https://plotly.com/python)
- [Pandas Documentation](https://pandas.pydata.org)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💼 Author

**Iiijul** - Default Loan Prediction Project
- GitHub: [@iiijul](https://github.com/iiijul)
- Repository: [default_loan_prediction](https://github.com/iiijul/default_loan_prediction)

## 📞 Contact

For questions, suggestions, or support:
- Create an issue on GitHub
- Email: contact@example.com

## 🎯 Future Enhancements

- [ ] Add more ML algorithms (Random Forest, XGBoost)
- [ ] Implement feature importance analysis
- [ ] Add model explainability (SHAP values)
- [ ] Create API endpoint for predictions
- [ ] Add multi-language support
- [ ] Implement user authentication
- [ ] Add real-time model retraining

## 📝 Changelog

### Version 1.0.0 (2024-07-19)
- Initial release
- Logistic Regression model implementation
- Interactive Streamlit dashboard
- Comprehensive data analytics
- Model performance metrics
- ROC curve analysis
- Feature correlation analysis

---

**Status:** ✅ Fully Functional | **Last Updated:** 2024-07-19 | **Version:** 1.0.0
