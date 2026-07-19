# 🚀 Quick Start Guide - Loan Prediction Dashboard

## Installation & Setup (2 minutes)

### Step 1: Navigate to Project Directory
```bash
cd /workspaces/default_loan_prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Dashboard
```bash
streamlit run app.py
```

### Step 4: Access the Dashboard
Open your browser and go to:
- **Local:** `http://localhost:8501`
- **Web:** Check the terminal output for the URL

---

## 🎯 Dashboard Navigation

### 🏠 Home
- Overview statistics
- Loan distribution analysis
- Quick metrics dashboard

### 🔮 Predictions
- **Input Fields:**
  - Principal Amount ($500-$1500)
  - Loan Terms (7, 15, 30 days)
  - Borrower Age (20-60 years)
  - Gender (Male/Female)
  - Education Level (High School/Bachelors/Masters)

- **Output:**
  - Default probability percentage
  - Risk level (Low/Medium/High)
  - Visual probability gauge
  - Risk assessment tips

### 📊 Analytics
Four analysis categories:
1. **Demographics** - Age & gender insights
2. **Finance** - Principal & terms analysis
3. **Trends** - Education level patterns
4. **Correlations** - Feature correlation matrix

### 📈 Model Performance
- Accuracy: 66.67%
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve
- Classification Report
- Model Specifications

### ℹ️ About
- Project overview
- Technical specifications
- Key achievements
- Feature list

---

## 🧪 Test Predictions

### Low-Risk Scenario (Should show Green ✅)
```
Principal: $700
Terms: 7 days
Age: 45 years
Gender: Female
Education: Masters
Expected: ~20% default probability
```

### High-Risk Scenario (Should show Red ⚠️)
```
Principal: $1400
Terms: 30 days
Age: 25 years
Gender: Male
Education: High School
Expected: ~75% default probability
```

### Medium-Risk Scenario
```
Principal: $1000
Terms: 15 days
Age: 35 years
Gender: Male
Education: Bachelors
Expected: ~50% default probability
```

---

## 📊 Key Features

✅ **Real-time Predictions** - Get instant default probability
✅ **Interactive Visualizations** - Plotly charts with hover details
✅ **Comprehensive Analytics** - Explore data patterns
✅ **Model Metrics** - Detailed performance evaluation
✅ **Responsive Design** - Works on desktop, tablet, mobile
✅ **Beautiful UI** - Custom styling and gradients
✅ **Risk Indicators** - Color-coded risk levels

---

## 🔧 Customization

### Change Color Scheme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#3498db"      # Change this
backgroundColor = "#f8f9fa"   # Or this
textColor = "#2c3e50"         # Or this
```

### Change Model
Replace `logistic_regression_model.pkl` with a new model:
```python
import joblib
joblib.dump(new_model, 'logistic_regression_model.pkl')
```

### Add More Features
Edit `app.py` and add new sections in the appropriate page

---

## 📱 Performance Tips

1. **First Load:** May take 5-10 seconds (model caching)
2. **Subsequent Loads:** Should be instant
3. **Predictions:** Real-time, <1 second
4. **Analytics:** Loads in 2-3 seconds

---

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Streamlit not found" | Run: `pip install streamlit` |
| "Port 8501 already in use" | Run: `streamlit run app.py --server.port 8502` |
| "Model file not found" | Ensure `.pkl` files are in project directory |
| "Data file not found" | Ensure CSV file is in project directory |
| "Slow performance" | Clear Streamlit cache: `rm -rf ~/.streamlit` |

---

## 📚 File Structure

```
default_loan_prediction/
├── app.py                              # Main Streamlit app (782 lines)
├── Loan_Predictive.ipynb               # EDA & model development
├── logistic_regression_model.pkl       # Trained model
├── model_features.pkl                  # Feature names
├── Loan payments data-selected...csv   # Dataset (500 rows)
├── requirements.txt                    # Dependencies
├── README.md                           # Full documentation
├── QUICKSTART.md                       # This file
└── .streamlit/
    └── config.toml                     # Streamlit configuration
```

---

## 🎓 What You Can Do

1. **Predict:** Enter loan details and get default probability
2. **Analyze:** Explore data patterns and relationships
3. **Evaluate:** Review model performance metrics
4. **Export:** Download data and visualizations
5. **Customize:** Modify styling and add features

---

## 🎯 Example Workflow

1. **Start Dashboard**
   ```bash
   streamlit run app.py
   ```

2. **Explore Home Page**
   - View overall statistics
   - Check loan distribution

3. **Make a Prediction**
   - Go to "Predictions" page
   - Enter loan parameters
   - See risk probability

4. **Analyze Data**
   - Go to "Analytics" page
   - Explore charts
   - Find patterns

5. **Check Model**
   - Go to "Model Performance"
   - Review metrics
   - Understand model behavior

---

## 💡 Tips for Best Results

✓ Start with the Home page for overview
✓ Use Analytics to understand data first
✓ Try different prediction scenarios
✓ Check Model Performance for confidence
✓ Use the About page for more info

---

## 📞 Need Help?

- Check README.md for detailed documentation
- Review model specifications in About page
- Test with sample predictions
- Check console for error messages

---

## 🎉 You're All Set!

Your Loan Prediction Dashboard is ready to use.

Start by running:
```bash
streamlit run app.py
```

Enjoy! 🚀
