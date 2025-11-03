import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Brain Cancer Data Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .dataframe {
        font-size: 0.9rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        color: #0c5460;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header"><h1>🧠 Brain Cancer Data Analysis Platform</h1><p style="margin:0; font-size: 1.1rem;">Advanced Machine Learning for Medical Data Prediction</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Navigation")
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.info("""
    This application analyzes brain cancer data using machine learning models to predict outcomes.
    
    **Features:**
    - Automated data preprocessing
    - Dual ML model comparison
    - Interactive visualizations
    - Performance metrics
    """)
    
    st.markdown("### 🔧 General Settings")
    random_state = st.number_input("Random State", value=100, min_value=0, max_value=1000, help="Set seed for reproducibility")
    train_size = st.slider("Training Data Size", min_value=0.1, max_value=0.9, value=0.33, step=0.05, help="Proportion of data used for training")
    cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5, help="Number of folds for cross-validation")
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Parameters")
    
    with st.expander("🔵 Linear Regression", expanded=False):
        lr_fit_intercept = st.checkbox("Fit Intercept", value=True, help="Calculate intercept for linear model")
    
    with st.expander("🌲 Gradient Boosting", expanded=False):
        gb_n_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=400, step=50, help="Number of boosting stages")
        gb_learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.15, 0.2], value=0.1, help="Step size shrinkage")
        gb_max_depth = st.slider("Max Depth", min_value=2, max_value=10, value=5, help="Maximum depth of trees")
        gb_min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, help="Minimum samples to split node")

st.markdown("## 📁 Step 1: Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Upload your Excel file containing brain cancer data",
    type=['xlsx', 'xls'],
    help="Upload an Excel file with brain cancer patient data"
)

if uploaded_file is None:
    default_file_path = "attached_assets/BrainCancerData_1762202235894.xlsx"
    try:
        uploaded_file = open(default_file_path, 'rb')
        st.markdown('<div class="info-box">ℹ️ Using default dataset from attached_assets folder</div>', unsafe_allow_html=True)
    except:
        st.warning("⚠️ Please upload an Excel file to begin analysis")
        st.stop()

if uploaded_file is not None:
    try:
        df_original = pd.read_excel(uploaded_file)
        
        st.markdown('<div class="success-box">✅ Dataset loaded successfully!</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df_original.shape[0])
        with col2:
            st.metric("Total Columns", df_original.shape[1])
        with col3:
            st.metric("Missing Values", df_original.isnull().sum().sum())
        
        st.markdown("### 📋 Raw Data Preview")
        st.dataframe(df_original.head(10), use_container_width=True)
        
        with st.expander("📊 Statistical Summary"):
            st.dataframe(df_original.describe(), use_container_width=True)
        
        st.markdown("---")
        st.markdown("## 🧹 Step 2: Data Preprocessing")
        
        df = df_original.copy()
        
        initial_rows = len(df)
        df = df.dropna(how='any')
        rows_after_cleaning = len(df)
        rows_removed = initial_rows - rows_after_cleaning
        
        st.markdown(f'<div class="info-box">Removed {rows_removed} rows with missing values. {rows_after_cleaning} rows remaining.</div>', unsafe_allow_html=True)
        
        columns_to_drop = []
        if 'No.' in df.columns:
            columns_to_drop.append('No.')
        if 'class label' in df.columns:
            columns_to_drop.append('class label')
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            st.success(f"✅ Removed non-feature columns: {', '.join(columns_to_drop)}")
        
        if 'v1' not in df.columns:
            st.error("❌ Error: Target column 'v1' not found in dataset. Please ensure your data contains a 'v1' column.")
            st.stop()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cleaned Rows", rows_after_cleaning)
        with col2:
            st.metric("Features", df.shape[1] - 1)
        
        with st.expander("🔍 View Cleaned Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.markdown("## 🤖 Step 3: Model Training & Evaluation")
        
        if st.button("🚀 Train Models", use_container_width=True):
            with st.spinner("Training machine learning models..."):
                y = df['v1'].copy()
                X = df.drop(columns=['v1'])
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    train_size=train_size, 
                    random_state=int(random_state)
                )
                
                st.markdown("### 📈 Model 1: Linear Regression")
                
                lr_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LinearRegression(fit_intercept=lr_fit_intercept))
                ])
                
                lr_pipeline.fit(X_train, y_train)
                lr_predictions = lr_pipeline.predict(X_test)
                
                lr_r2 = r2_score(y_test, lr_predictions)
                lr_mae = mean_absolute_error(y_test, lr_predictions)
                lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score (Test)", f"{lr_r2:.4f}", help="Coefficient of determination (higher is better)")
                with col2:
                    st.metric("MAE (Test)", f"{lr_mae:.4f}", help="Mean Absolute Error (lower is better)")
                with col3:
                    st.metric("RMSE (Test)", f"{lr_rmse:.4f}", help="Root Mean Squared Error (lower is better)")
                
                with st.spinner("Running cross-validation..."):
                    cv_scores_lr = cross_validate(
                        lr_pipeline, X, y, 
                        cv=cv_folds, 
                        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
                        return_train_score=True
                    )
                    
                    lr_cv_r2_mean = cv_scores_lr['test_r2'].mean()
                    lr_cv_r2_std = cv_scores_lr['test_r2'].std()
                    lr_cv_mae_mean = -cv_scores_lr['test_neg_mean_absolute_error'].mean()
                    lr_cv_rmse_mean = -cv_scores_lr['test_neg_root_mean_squared_error'].mean()
                
                with st.expander("📊 Cross-Validation Results (Linear Regression)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("CV R² Score", f"{lr_cv_r2_mean:.4f}", delta=f"±{lr_cv_r2_std:.4f}")
                    with col2:
                        st.metric("CV MAE", f"{lr_cv_mae_mean:.4f}")
                    with col3:
                        st.metric("CV RMSE", f"{lr_cv_rmse_mean:.4f}")
                    
                    cv_df_lr = pd.DataFrame({
                        'Fold': range(1, cv_folds + 1),
                        'R² Score': cv_scores_lr['test_r2'],
                        'MAE': -cv_scores_lr['test_neg_mean_absolute_error'],
                        'RMSE': -cv_scores_lr['test_neg_root_mean_squared_error']
                    })
                    st.dataframe(cv_df_lr, use_container_width=True)
                
                st.markdown("### 🌲 Model 2: Gradient Boosting Regressor")
                
                gb_model = GradientBoostingRegressor(
                    n_estimators=gb_n_estimators,
                    max_depth=gb_max_depth,
                    min_samples_split=gb_min_samples_split,
                    learning_rate=gb_learning_rate,
                    random_state=int(random_state)
                )
                gb_model.fit(X_train, y_train)
                gb_predictions = gb_model.predict(X_test)
                
                gb_r2 = r2_score(y_test, gb_predictions)
                gb_mae = mean_absolute_error(y_test, gb_predictions)
                gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score (Test)", f"{gb_r2:.4f}", help="Coefficient of determination (higher is better)")
                with col2:
                    st.metric("MAE (Test)", f"{gb_mae:.4f}", help="Mean Absolute Error (lower is better)")
                with col3:
                    st.metric("RMSE (Test)", f"{gb_rmse:.4f}", help="Root Mean Squared Error (lower is better)")
                
                with st.spinner("Running cross-validation..."):
                    cv_scores_gb = cross_validate(
                        gb_model, X, y, 
                        cv=cv_folds, 
                        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
                        return_train_score=True
                    )
                    
                    gb_cv_r2_mean = cv_scores_gb['test_r2'].mean()
                    gb_cv_r2_std = cv_scores_gb['test_r2'].std()
                    gb_cv_mae_mean = -cv_scores_gb['test_neg_mean_absolute_error'].mean()
                    gb_cv_rmse_mean = -cv_scores_gb['test_neg_root_mean_squared_error'].mean()
                
                with st.expander("📊 Cross-Validation Results (Gradient Boosting)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("CV R² Score", f"{gb_cv_r2_mean:.4f}", delta=f"±{gb_cv_r2_std:.4f}")
                    with col2:
                        st.metric("CV MAE", f"{gb_cv_mae_mean:.4f}")
                    with col3:
                        st.metric("CV RMSE", f"{gb_cv_rmse_mean:.4f}")
                    
                    cv_df_gb = pd.DataFrame({
                        'Fold': range(1, cv_folds + 1),
                        'R² Score': cv_scores_gb['test_r2'],
                        'MAE': -cv_scores_gb['test_neg_mean_absolute_error'],
                        'RMSE': -cv_scores_gb['test_neg_root_mean_squared_error']
                    })
                    st.dataframe(cv_df_gb, use_container_width=True)
                
                st.markdown("---")
                st.markdown("## 📊 Step 4: Model Comparison & Visualization")
                
                comparison_data = pd.DataFrame({
                    'Model': ['Linear Regression', 'Gradient Boosting'],
                    'Test R² Score': [lr_r2, gb_r2],
                    'Test MAE': [lr_mae, gb_mae],
                    'Test RMSE': [lr_rmse, gb_rmse],
                    'CV R² Score': [lr_cv_r2_mean, gb_cv_r2_mean],
                    'CV MAE': [lr_cv_mae_mean, gb_cv_mae_mean],
                    'CV RMSE': [lr_cv_rmse_mean, gb_cv_rmse_mean]
                })
                
                st.markdown("### 🏆 Comprehensive Performance Comparison")
                st.dataframe(comparison_data, use_container_width=True)
                
                best_model = "Gradient Boosting" if gb_r2 > lr_r2 else "Linear Regression"
                improvement = abs(gb_r2 - lr_r2) * 100
                
                st.markdown(f'<div class="success-box">🏅 <strong>Best Performing Model:</strong> {best_model} (Test R² improvement: {improvement:.2f}%)</div>', unsafe_allow_html=True)
                
                st.markdown("### 📊 Cross-Validation Score Distribution")
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                cv_metrics = [
                    ('R² Score', cv_scores_lr['test_r2'], cv_scores_gb['test_r2']),
                    ('MAE', -cv_scores_lr['test_neg_mean_absolute_error'], -cv_scores_gb['test_neg_mean_absolute_error']),
                    ('RMSE', -cv_scores_lr['test_neg_root_mean_squared_error'], -cv_scores_gb['test_neg_root_mean_squared_error'])
                ]
                
                for idx, (metric_name, lr_vals, gb_vals) in enumerate(cv_metrics):
                    x_pos = np.arange(cv_folds)
                    width = 0.35
                    
                    axes[idx].bar(x_pos - width/2, lr_vals, width, label='Linear Regression', color='#667eea', alpha=0.8, edgecolor='white', linewidth=1)
                    axes[idx].bar(x_pos + width/2, gb_vals, width, label='Gradient Boosting', color='#764ba2', alpha=0.8, edgecolor='white', linewidth=1)
                    
                    axes[idx].set_xlabel('Fold', fontsize=10, fontweight='bold')
                    axes[idx].set_ylabel(metric_name, fontsize=10, fontweight='bold')
                    axes[idx].set_title(f'{metric_name} Across Folds', fontsize=11, fontweight='bold', pad=10)
                    axes[idx].set_xticks(x_pos)
                    axes[idx].set_xticklabels([f'{i+1}' for i in range(cv_folds)])
                    axes[idx].legend(fontsize=8)
                    axes[idx].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown("### 📉 Prediction Accuracy Visualizations")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].scatter(lr_predictions, y_test, alpha=0.6, color='#667eea', edgecolors='white', linewidth=0.5)
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
                axes[0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
                axes[0].set_ylabel('Actual Values', fontsize=11, fontweight='bold')
                axes[0].set_title(f'Linear Regression\nR² = {lr_r2:.4f}', fontsize=12, fontweight='bold', pad=15)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].scatter(gb_predictions, y_test, alpha=0.6, color='#764ba2', edgecolors='white', linewidth=0.5)
                axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
                axes[1].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
                axes[1].set_ylabel('Actual Values', fontsize=11, fontweight='bold')
                axes[1].set_title(f'Gradient Boosting\nR² = {gb_r2:.4f}', fontsize=12, fontweight='bold', pad=15)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown("### 📊 Model Performance Metrics")
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                metrics = ['R² Score', 'MAE', 'RMSE']
                lr_values = [lr_r2, lr_mae, lr_rmse]
                gb_values = [gb_r2, gb_mae, gb_rmse]
                
                for idx, (metric, lr_val, gb_val) in enumerate(zip(metrics, lr_values, gb_values)):
                    x = np.arange(2)
                    width = 0.6
                    bars = axes[idx].bar(x, [lr_val, gb_val], width, color=['#667eea', '#764ba2'], alpha=0.8, edgecolor='white', linewidth=1.5)
                    axes[idx].set_ylabel(metric, fontsize=11, fontweight='bold')
                    axes[idx].set_xticks(x)
                    axes[idx].set_xticklabels(['Linear\nRegression', 'Gradient\nBoosting'], fontsize=9)
                    axes[idx].set_title(metric, fontsize=12, fontweight='bold', pad=10)
                    axes[idx].grid(True, alpha=0.3, axis='y')
                    
                    for bar in bars:
                        height = bar.get_height()
                        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                     f'{height:.4f}',
                                     ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown("### 🎯 Feature Importance (Gradient Boosting)")
                
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': gb_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
                bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors, edgecolor='white', linewidth=1.5)
                ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
                ax.set_ylabel('Features', fontsize=11, fontweight='bold')
                ax.set_title('Top 10 Most Important Features', fontsize=13, fontweight='bold', pad=15)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.markdown("---")
                st.markdown("## 📥 Download Analysis Report")
                
                report_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Brain Cancer Data Analysis Report</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 40px;
                            background-color: #f5f5f5;
                        }}
                        .header {{
                            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 30px;
                            border-radius: 10px;
                            margin-bottom: 30px;
                        }}
                        .section {{
                            background: white;
                            padding: 20px;
                            margin-bottom: 20px;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }}
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin: 20px 0;
                        }}
                        th, td {{
                            padding: 12px;
                            text-align: left;
                            border-bottom: 1px solid #ddd;
                        }}
                        th {{
                            background-color: #667eea;
                            color: white;
                        }}
                        .metric {{
                            display: inline-block;
                            background: #f0f0f0;
                            padding: 15px 25px;
                            margin: 10px;
                            border-radius: 5px;
                            border-left: 4px solid #667eea;
                        }}
                        .metric-value {{
                            font-size: 24px;
                            font-weight: bold;
                            color: #333;
                        }}
                        .metric-label {{
                            font-size: 14px;
                            color: #666;
                        }}
                        .best-model {{
                            background: #d4edda;
                            border: 1px solid #c3e6cb;
                            color: #155724;
                            padding: 15px;
                            border-radius: 5px;
                            margin: 20px 0;
                        }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>🧠 Brain Cancer Data Analysis Report</h1>
                        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="section">
                        <h2>📊 Dataset Summary</h2>
                        <div class="metric">
                            <div class="metric-label">Total Rows</div>
                            <div class="metric-value">{df_original.shape[0]}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Features</div>
                            <div class="metric-value">{df.shape[1] - 1}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Cleaned Rows</div>
                            <div class="metric-value">{rows_after_cleaning}</div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>🏆 Model Performance Comparison</h2>
                        <div class="best-model">
                            <strong>🏅 Best Performing Model:</strong> {best_model}<br>
                            <strong>R² Score Improvement:</strong> {improvement:.2f}%
                        </div>
                        {comparison_data.to_html(index=False, classes='styled-table')}
                    </div>
                    
                    <div class="section">
                        <h2>📈 Linear Regression Results</h2>
                        <h3>Test Set Performance</h3>
                        <div class="metric">
                            <div class="metric-label">R² Score</div>
                            <div class="metric-value">{lr_r2:.4f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">MAE</div>
                            <div class="metric-value">{lr_mae:.4f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">RMSE</div>
                            <div class="metric-value">{lr_rmse:.4f}</div>
                        </div>
                        
                        <h3>Cross-Validation Results</h3>
                        <div class="metric">
                            <div class="metric-label">CV R² Score</div>
                            <div class="metric-value">{lr_cv_r2_mean:.4f} ± {lr_cv_r2_std:.4f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">CV MAE</div>
                            <div class="metric-value">{lr_cv_mae_mean:.4f}</div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>🌲 Gradient Boosting Results</h2>
                        <h3>Model Parameters</h3>
                        <ul>
                            <li><strong>Number of Estimators:</strong> {gb_n_estimators}</li>
                            <li><strong>Learning Rate:</strong> {gb_learning_rate}</li>
                            <li><strong>Max Depth:</strong> {gb_max_depth}</li>
                            <li><strong>Min Samples Split:</strong> {gb_min_samples_split}</li>
                        </ul>
                        
                        <h3>Test Set Performance</h3>
                        <div class="metric">
                            <div class="metric-label">R² Score</div>
                            <div class="metric-value">{gb_r2:.4f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">MAE</div>
                            <div class="metric-value">{gb_mae:.4f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">RMSE</div>
                            <div class="metric-value">{gb_rmse:.4f}</div>
                        </div>
                        
                        <h3>Cross-Validation Results</h3>
                        <div class="metric">
                            <div class="metric-label">CV R² Score</div>
                            <div class="metric-value">{gb_cv_r2_mean:.4f} ± {gb_cv_r2_std:.4f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">CV MAE</div>
                            <div class="metric-value">{gb_cv_mae_mean:.4f}</div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>🎯 Top 10 Feature Importance (Gradient Boosting)</h2>
                        {feature_importance.to_html(index=False)}
                    </div>
                    
                    <div class="section">
                        <h2>⚙️ Configuration</h2>
                        <ul>
                            <li><strong>Random State:</strong> {random_state}</li>
                            <li><strong>Training Size:</strong> {train_size * 100:.0f}%</li>
                            <li><strong>Cross-Validation Folds:</strong> {cv_folds}</li>
                        </ul>
                    </div>
                </body>
                </html>
                """
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📄 Download HTML Report",
                        data=report_html,
                        file_name=f"brain_cancer_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                
                with col2:
                    csv_data = comparison_data.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Performance Data (CSV)",
                        data=csv_data,
                        file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.success("✅ Analysis complete! Models have been trained and evaluated successfully.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your Excel file has the correct format with a 'v1' column for the target variable.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='margin: 0;'>🧠 Brain Cancer Data Analysis Platform | Built with Streamlit</p>
    <p style='margin: 0; font-size: 0.9rem;'>Powered by scikit-learn & advanced machine learning algorithms</p>
</div>
""", unsafe_allow_html=True)
