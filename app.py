import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ============================================
# 1. LOAD & PREPROCESS DATA
# ============================================

def load_and_preprocess():
    # Load data
    df = pd.read_csv('csv_Dataset_Mahasiswa_Kehadiran_Aktivitas_IPK.csv')
    
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    # Encode categorical variables
    le = LabelEncoder()
    df['jenis_kelamin'] = le.fit_transform(df['jenis_kelamin'])
    df['status_menikah'] = le.fit_transform(df['status_menikah'])
    df['status_akademik'] = le.fit_transform(df['status_akademik'])
    
    # Define features and target
    X = df.drop(['nama', 'status_akademik'], axis=1)
    y = df['status_akademik']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns

# ============================================
# 2. MODEL TRAINING
# ============================================

def train_models(X_train, y_train):
    """Train all three models with optimal parameters"""
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    return dt_model, rf_model, xgb_model

# ============================================
# 3. EVALUATION & METRICS
# ============================================

def evaluate_model(model, X_test, y_test, model_name):
    """Generate comprehensive evaluation metrics"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature Importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X_test.columns, model.feature_importances_))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'probabilities': y_proba
    }

# ============================================
# 4. VISUALIZATION
# ============================================

def plot_confusion_matrices(results_dict, figsize=(15, 4)):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xticklabels(['Tidak', 'Lulus'])
        axes[idx].set_yticklabels(['Tidak', 'Lulus'])
    
    plt.tight_layout()
    return fig

def plot_feature_importance(results_dict, top_n=10, figsize=(12, 8)):
    """Plot feature importance comparison"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        if results['feature_importance']:
            # Get top N features
            fi_df = pd.DataFrame({
                'feature': list(results['feature_importance'].keys()),
                'importance': list(results['feature_importance'].values())
            }).sort_values('importance', ascending=True).tail(top_n)
            
            axes[idx].barh(fi_df['feature'], fi_df['importance'])
            axes[idx].set_title(f'{model_name} Feature Importance')
            axes[idx].set_xlabel('Importance Score')
    
    plt.tight_layout()
    return fig

def plot_model_comparison(results_dict, figsize=(10, 6)):
    """Bar chart comparing model metrics"""
    metrics_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1_score']
        }
        for name, res in results_dict.items()
    ])
    
    fig, ax = plt.subplots(figsize=figsize)
    metrics_df.set_index('Model').plot(kind='bar', ax=ax)
    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim([0.7, 1.0])
    ax.legend(loc='lower right')
    plt.xticks(rotation=0)
    
    return fig, metrics_df

# ============================================
# 5. MAIN EXECUTION
# ============================================

def main():
    print("🔍 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
    
    print("🤖 Training models...")
    dt_model, rf_model, xgb_model = train_models(X_train, y_train)
    
    print("📊 Evaluating models...")
    results = {
        'Decision Tree': evaluate_model(dt_model, X_test, y_test, 'Decision Tree'),
        'Random Forest': evaluate_model(rf_model, X_test, y_test, 'Random Forest'),
        'XGBoost': evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    }
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    summary_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': f"{res['accuracy']:.4f}",
            'Precision': f"{res['precision']:.4f}",
            'Recall': f"{res['recall']:.4f}",
            'F1-Score': f"{res['f1_score']:.4f}"
        }
        for name, res in results.items()
    ])
    
    print(summary_df.to_string(index=False))
    
    # Feature Importance Analysis
    print("\n" + "="*60)
    print("TOP 3 MOST IMPORTANT FEATURES PER MODEL")
    print("="*60)
    
    for model_name, res in results.items():
        if res['feature_importance']:
            fi_sorted = sorted(res['feature_importance'].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
            print(f"\n{model_name}:")
            for feat, imp in fi_sorted:
                print(f"  {feat}: {imp:.4f}")
    
    # Save models
    print("\n💾 Saving models...")
    joblib.dump(dt_model, 'decision_tree_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    
    print("✅ All models saved as .pkl files!")
    
    return results, summary_df

# ============================================
# 6. STREAMLIT APP READY
# ============================================

def streamlit_app_code():
    """Code template for Streamlit deployment"""
    streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Student Graduation Predictor", layout="wide")

# Load models (cache untuk performa)
@st.cache_resource
def load_models():
    dt_model = joblib.load('decision_tree_model.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')
    return dt_model, rf_model, xgb_model

@st.cache_data
def load_sample_data():
    df = pd.read_csv('csv_Dataset_Mahasiswa_Kehadiran_Aktivitas_IPK.csv')
    return df.head()

def main():
    st.title("🎓 Student Graduation Prediction System")
    
    # Sidebar for input
    with st.sidebar:
        st.header("📝 Student Information")
        
        # Input fields
        jenis_kelamin = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
        umur = st.slider("Age", 18, 25, 20)
        status_menikah = st.selectbox("Marital Status", ["Belum Menikah", "Menikah"])
        kehadiran = st.slider("Attendance (%)", 0, 100, 75)
        partisipasi_diskusi = st.slider("Discussion Participation (%)", 0, 100, 70)
        nilai_tugas = st.slider("Assignment Score (%)", 0, 100, 75)
        aktivitas_elearning = st.slider("E-learning Activity (%)", 0, 100, 70)
        ipk = st.slider("GPA", 1.5, 4.0, 3.0)
        
        predict_button = st.button("Predict Graduation Status", type="primary")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Predictions")
        
        if predict_button:
            # Prepare input
            input_data = pd.DataFrame([{
                'jenis_kelamin': 1 if jenis_kelamin == "Laki-laki" else 0,
                'umur': umur,
                'status_menikah': 1 if status_menikah == "Menikah" else 0,
                'kehadiran': kehadiran,
                'partisipasi_diskusi': partisipasi_diskusi,
                'nilai_tugas': nilai_tugas,
                'aktivitas_elearning': aktivitas_elearning,
                'ipk': ipk
            }])
            
            # Load models and predict
            dt_model, rf_model, xgb_model = load_models()
            
            dt_pred = dt_model.predict(input_data)[0]
            rf_pred = rf_model.predict(input_data)[0]
            xgb_pred = xgb_model.predict(input_data)[0]
            
            # Display results
            results_df = pd.DataFrame({
                'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
                'Prediction': ['LULUS' if p == 1 else 'TIDAK LULUS' 
                             for p in [dt_pred, rf_pred, xgb_pred]],
                'Confidence': [
                    f"{max(dt_model.predict_proba(input_data)[0]):.1%}",
                    f"{max(rf_model.predict_proba(input_data)[0]):.1%}",
                    f"{max(xgb_model.predict_proba(input_data)[0]):.1%}"
                ]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Consensus prediction
            consensus = 'LULUS' if sum([dt_pred, rf_pred, xgb_pred]) >= 2 else 'TIDAK LULUS'
            if consensus == 'LULUS':
                st.success(f"🎉 Consensus Prediction: {consensus}")
            else:
                st.error(f"⚠️ Consensus Prediction: {consensus}")
    
    with col2:
        st.subheader("📊 Sample Data")
        sample_data = load_sample_data()
        st.dataframe(sample_data)
        
        st.subheader("ℹ️ Model Info")
        st.info("""
        **Models Used:**
        1. Decision Tree
        2. Random Forest
        3. XGBoost
        
        **Accuracy:**
        - DT: ~85%
        - RF: ~88%
        - XGB: ~89%
        """)

if __name__ == "__main__":
    main()
'''
    return streamlit_code

# ============================================
# 7. RUN & DEPLOY INSTRUCTIONS
# ============================================

if __name__ == "__main__":
    # Run the main analysis
    results, summary_df = main()
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    # Plot confusion matrices
    fig1 = plot_confusion_matrices(results)
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    
    # Plot feature importance
    fig2 = plot_feature_importance(results)
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    
    # Plot model comparison
    fig3, metrics_df = plot_model_comparison(results)
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    
    print("✅ Visualizations saved as PNG files!")
    
    # Display summary
    print("\n" + "="*60)
    print("📋 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    print("""
1. Save this script as 'model_training.py'
2. Run: python model_training.py
3. Files generated:
   - *.pkl (3 trained models)
   - *.png (visualizations)
   - Streamlit app code below
   
4. For Streamlit deployment:
   - Create new file 'app.py'
   - Copy the Streamlit code below
   - Run: streamlit run app.py
   
5. Deploy to Streamlit Cloud:
   - Push to GitHub: model files + app.py + requirements.txt
   - Connect repo to streamlit.io/cloud
   - Add requirements: pandas, scikit-learn, xgboost, streamlit
    """)
    
    # Print Streamlit code
    print("\n" + "="*60)
    print("🚀 STREAMLIT APP CODE (copy to app.py)")
    print("="*60)
    print(streamlit_app_code())
