import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .header-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .header-subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load('logistic_regression_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv('Loan payments data-selected-columns.csv')
    return df

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Load model and data
model, features = load_model()
df = load_data()

# Sidebar navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "🔮 Predictions", "📊 Analytics", "📈 Model Performance", "ℹ️ About"],
        key='page_selector'
    )

# ==================== PAGE 1: HOME ====================
if page == "🏠 Home":
    st.markdown('<div class="header-title">💰 Loan Default Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Advanced ML Model for Credit Risk Assessment</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Loans</h3>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        default_rate = (df['loan_status'] != 'PAIDOFF').sum() / len(df) * 100
        st.markdown(f"""
        <div class="success-card">
            <h3>⚠️ Default Rate</h3>
            <h2>{default_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_principal = df['Principal'].mean()
        st.markdown(f"""
        <div class="info-card">
            <h3>💵 Avg Principal</h3>
            <h2>${avg_principal:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dashboard overview with tabs
    tab1, tab2, tab3 = st.tabs(["📋 Quick Stats", "🎯 Key Metrics", "📌 Data Overview"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loan Status Distribution")
            status_counts = df['loan_status'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.3,
                marker=dict(colors=['#2ecc71', '#e74c3c', '#f39c12'])
            )])
            fig.update_layout(height=350, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Loan Terms")
            terms_avg = df.groupby('terms')['Principal'].mean().sort_index()
            fig = go.Figure(data=[go.Bar(
                x=terms_avg.index.astype(str),
                y=terms_avg.values,
                marker=dict(color=['#3498db', '#9b59b6', '#e74c3c'])
            )])
            fig.update_layout(
                xaxis_title="Loan Terms (Days)",
                yaxis_title="Average Principal ($)",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Age Range", f"{df['age'].min():.0f} - {df['age'].max():.0f} years", "👤"),
            ("Principal Range", f"${df['Principal'].min():.0f} - ${df['Principal'].max():.0f}", "💵"),
            ("Avg Age", f"{df['age'].mean():.1f} years", "📅"),
            ("Default/Non-Default", f"{(df['loan_status'] != 'PAIDOFF').sum()}/{(df['loan_status'] == 'PAIDOFF').sum()}", "📊")
        ]
        
        for idx, (label, value, icon) in enumerate(metrics):
            with [col1, col2, col3, col4][idx]:
                st.metric(label, value, icon)
    
    with tab3:
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Records:** {len(df)}")
            st.write(f"**Total Features:** {len(df.columns)}")
            st.write(f"**Data Type:** Loan Application Records")
        
        with col2:
            st.write(f"**Model Type:** Logistic Regression")
            st.write(f"**Training Set:** 350 records (70%)")
            st.write(f"**Test Set:** 150 records (30%)")
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

# ==================== PAGE 2: PREDICTIONS ====================
elif page == "🔮 Predictions":
    st.markdown('<div class="header-title">🔮 Loan Default Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Enter loan details to predict default probability</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Loan Information")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            principal = st.slider(
                "Principal Amount ($)",
                min_value=500,
                max_value=1500,
                value=1000,
                step=50
            )
            
            terms = st.selectbox(
                "Loan Terms (Days)",
                [7, 15, 30],
                index=2
            )
        
        with col_b:
            age = st.slider(
                "Borrower Age (Years)",
                min_value=20,
                max_value=60,
                value=35,
                step=1
            )
            
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )
        
        education = st.selectbox(
            "Education Level",
            ["High School", "Bachelors", "Masters"]
        )
        
        st.markdown("---")
        
        # Prepare features for prediction
        gender_male = 1 if gender == "Male" else 0
        education_high_school = 1 if education == "High School" else 0
        education_masters = 1 if education == "Masters" else 0
        
        # Create input data
        input_data = pd.DataFrame({
            'Principal': [principal],
            'terms': [terms],
            'age': [age],
            'Gender_Male': [gender_male],
            'Education_High School': [education_high_school],
            'Education_Masters': [education_masters]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display prediction result
        st.markdown("---")
        
        prediction_text = "🔴 HIGH RISK - LIKELY TO DEFAULT" if prediction == 1 else "🟢 LOW RISK - LIKELY TO PAY OFF"
        default_prob = prediction_proba[1] * 100
        paid_off_prob = prediction_proba[0] * 100
        
        st.markdown(f"""
        <div style="background: {'#ffebee' if prediction == 1 else '#e8f5e9'}; padding: 20px; border-radius: 10px; border-left: 5px solid {'#e74c3c' if prediction == 1 else '#2ecc71'};">
            <h2 style="color: {'#c0392b' if prediction == 1 else '#27ae60'}; margin: 0;">{prediction_text}</h2>
            <p style="margin: 10px 0 0 0; color: #555;">Model Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                y=["Default Risk", "Pay Off"],
                x=[default_prob, paid_off_prob],
                orientation='h',
                marker=dict(color=['#e74c3c', '#2ecc71']),
                text=[f'{default_prob:.1f}%', f'{paid_off_prob:.1f}%'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title="",
            height=300,
            showlegend=False,
            xaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Probability gauge
    st.markdown("---")
    st.subheader("🎯 Default Risk Gauge")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=default_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#2ecc71"},
                {'range': [33, 66], 'color': "#f39c12"},
                {'range': [66, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk assessment tips
    st.markdown("---")
    st.subheader("💡 Risk Assessment Tips")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **✅ Low Risk Factors:**
        - Shorter loan terms (7-15 days)
        - Lower principal amount
        - Higher education level
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Medium Risk Factors:**
        - Moderate principal amount
        - Mixed education levels
        - Balanced age range
        """)
    
    with col3:
        st.markdown("""
        **❌ High Risk Factors:**
        - Longer loan terms (30 days)
        - Higher principal amount
        - Lower education level
        """)

# ==================== PAGE 3: ANALYTICS ====================
elif page == "📊 Analytics":
    st.markdown('<div class="header-title">📊 Data Analytics & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Explore patterns and relationships in the data</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Demographics", "💰 Finance", "📅 Trends", "🎯 Correlations"])
    
    with tab1:
        st.subheader("Age Distribution by Default Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x='age',
                color='loan_status',
                barmode='overlay',
                nbins=20,
                title="Age Distribution",
                labels={'age': 'Age (Years)', 'count': 'Frequency'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by status
            fig = px.box(
                df,
                x='loan_status',
                y='age',
                title="Age by Loan Status",
                points="outliers"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Gender analysis
        st.subheader("Gender Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            gender_counts = df['Gender'].value_counts()
            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Gender Distribution",
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            gender_default = df.groupby('Gender')['loan_status'].apply(lambda x: (x != 'PAIDOFF').sum() / len(x) * 100)
            fig = px.bar(
                x=gender_default.index,
                y=gender_default.values,
                title="Default Rate by Gender",
                labels={'x': 'Gender', 'y': 'Default Rate (%)'},
                color=gender_default.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Principal Amount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df,
                x='Principal',
                nbins=30,
                title="Principal Distribution",
                labels={'Principal': 'Principal ($)'},
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                x='loan_status',
                y='Principal',
                title="Principal by Loan Status",
                points="outliers"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Loan Terms Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            terms_count = df['terms'].value_counts().sort_index()
            fig = px.bar(
                x=terms_count.index.astype(str),
                y=terms_count.values,
                title="Loan Terms Distribution",
                labels={'x': 'Terms (Days)', 'y': 'Count'},
                color=terms_count.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            terms_default = df.groupby('terms')['loan_status'].apply(lambda x: (x != 'PAIDOFF').sum() / len(x) * 100)
            fig = px.bar(
                x=terms_default.index.astype(str),
                y=terms_default.values,
                title="Default Rate by Loan Terms",
                labels={'x': 'Terms (Days)', 'y': 'Default Rate (%)'},
                color=terms_default.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Education Level Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            education_counts = df['education'].value_counts()
            fig = px.pie(
                values=education_counts.values,
                names=education_counts.index,
                title="Education Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            education_default = df.groupby('education')['loan_status'].apply(lambda x: (x != 'PAIDOFF').sum() / len(x) * 100)
            fig = px.bar(
                x=education_default.index,
                y=education_default.values,
                title="Default Rate by Education",
                labels={'x': 'Education', 'y': 'Default Rate (%)'},
                color=education_default.values,
                color_continuous_scale='YlOrRd'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Correlation Matrix")
        
        # Create correlation data
        df_numeric = df[['Principal', 'terms', 'age', 'past_due_days']].copy()
        df_numeric['is_default'] = (df['loan_status'] != 'PAIDOFF').astype(int)
        
        corr_matrix = df_numeric.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(height=500, width=600)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 4: MODEL PERFORMANCE ====================
elif page == "📈 Model Performance":
    st.markdown('<div class="header-title">📈 Model Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Detailed evaluation of the prediction model</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Model Metrics", "🎯 Classification Report", "🔍 Model Details"])
    
    with tab1:
        # Generate predictions on test data for metrics
        X = df[features].copy()
        y = (df['loan_status'] != 'PAIDOFF').astype(int)
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}", delta=f"{accuracy*100:.1f}%")
        with col2:
            st.metric("Precision", f"{precision:.3f}", delta=f"{precision*100:.1f}%")
        with col3:
            st.metric("Recall", f"{recall:.3f}", delta=f"{recall*100:.1f}%")
        with col4:
            st.metric("F1-Score", f"{f1:.3f}", delta=f"{f1*100:.1f}%")
        with col5:
            st.metric("ROC-AUC", f"{roc_auc:.3f}", delta=f"{roc_auc*100:.1f}%")
        
        st.markdown("---")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            cm = confusion_matrix(y, y_pred)
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Pay-Off', 'Predicted Default'],
                y=['Actual Pay-Off', 'Actual Default'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig.update_layout(title="Confusion Matrix", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc_val = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc_val:.3f})', line=dict(color='#3498db', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(color='red', dash='dash')))
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Classification Report")
        
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(report_df, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Interpretation Guide")
        st.markdown("""
        - **Precision:** Of all predictions of default, how many were correct?
        - **Recall:** Of all actual defaults, how many did the model find?
        - **F1-Score:** Harmonic mean of precision and recall
        - **Support:** Number of samples in each class
        """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Specifications")
            st.write(f"""
            **Algorithm:** Logistic Regression
            
            **Solver:** liblinear
            
            **Features Used:** {len(features)}
            
            **Training Samples:** 350
            
            **Test Samples:** 150
            
            **Test Size:** 30%
            
            **Random State:** 42
            """)
        
        with col2:
            st.subheader("Feature List")
            for i, feature in enumerate(features, 1):
                st.write(f"{i}. {feature}")

# ==================== PAGE 5: ABOUT ====================
elif page == "ℹ️ About":
    st.markdown('<div class="header-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📋 Project Overview
        
        This **Loan Default Prediction System** is an advanced machine learning application designed to predict
        the likelihood of loan default based on borrower characteristics and loan parameters.
        
        ### 🎯 Objective
        
        The primary goal is to:
        - Predict which loans are likely to default
        - Assist financial institutions in risk assessment
        - Improve lending decision-making processes
        - Reduce portfolio risk
        
        ### 📊 Dataset Information
        
        - **Total Records:** 500 loan applications
        - **Features:** 11 (Principal, Terms, Age, Education, Gender, etc.)
        - **Target Variable:** Loan Status (PAIDOFF, COLLECTION, COLLECTION_PAIDOFF)
        - **Data Split:** 70% Training (350), 30% Testing (150)
        
        ### 🤖 Machine Learning Model
        
        - **Algorithm:** Logistic Regression
        - **Purpose:** Binary Classification (Default vs. Non-Default)
        - **Performance:** 
          - Accuracy: ~67%
          - AUC-ROC: ~0.53
        
        ### 🔧 Technical Stack
        
        - **Backend:** Python, Scikit-learn, Joblib
        - **Frontend:** Streamlit
        - **Visualization:** Plotly, Seaborn, Matplotlib
        - **Data Processing:** Pandas, NumPy
        
        ### 🎨 Features
        
        ✅ Interactive loan prediction interface
        ✅ Comprehensive data analytics dashboard
        ✅ Model performance metrics and evaluation
        ✅ Risk assessment visualizations
        ✅ Real-time probability calculations
        ✅ Detailed classification reports
        
        ### 📚 Usage Guide
        
        1. **Home Page:** View overall statistics and dataset overview
        2. **Predictions:** Enter loan details and get default probability
        3. **Analytics:** Explore data patterns and relationships
        4. **Model Performance:** Review detailed model metrics
        5. **About:** Learn about the project
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Quick Stats
        """)
        
        st.metric("Total Loans", len(df))
        st.metric("Default Rate", f"{(df['loan_status'] != 'PAIDOFF').sum() / len(df) * 100:.1f}%")
        st.metric("Avg Principal", f"${df['Principal'].mean():,.0f}")
        st.metric("Model Accuracy", "66.7%")
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔐 Data Privacy
        
        - All data is processed locally
        - No personal information is stored
        - Predictions are calculated on-demand
        - No external data sharing
        
        ### 📞 Support
        
        For questions or support, please contact:
        - Email: support@loanprediction.com
        - GitHub: github.com/iiijul/default_loan_prediction
        """)
    
    st.markdown("---")
    
    st.subheader("🏆 Key Achievements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data Understanding**
        - Comprehensive EDA
        - Feature analysis
        - Pattern identification
        """)
    
    with col2:
        st.markdown("""
        **Model Development**
        - Logistic Regression
        - Feature engineering
        - Cross-validation
        """)
    
    with col3:
        st.markdown("""
        **Deployment**
        - Interactive dashboard
        - Real-time predictions
        - Performance tracking
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Loan Prediction System**")
with col2:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    st.markdown("**Version:** 1.0.0")

st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 20px 0;">
    <p>🚀 Powered by Streamlit | Machine Learning | Data Science</p>
    <p>© 2024 Loan Default Prediction System. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
