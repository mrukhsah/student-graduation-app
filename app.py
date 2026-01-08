import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

# Page config
st.set_page_config(
    page_title="Student Graduation Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 Student Graduation Prediction System")
st.markdown("Predict whether a student will graduate based on academic performance")

# Initialize session state for models
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.models = {}

# Function to train models
@st.cache_resource
def train_models():
    """Train all models and cache them"""
    try:
        # Load data
        df = pd.read_csv('csv_Dataset_Mahasiswa_Kehadiran_Aktivitas_IPK.csv')
        
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
        
        # Train Decision Tree
        dt_model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        dt_model.fit(X_train, y_train)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Train XGBoost
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        
        return {
            'dt_model': dt_model,
            'rf_model': rf_model,
            'xgb_model': xgb_model,
            'accuracy': {
                'dt': dt_model.score(X_test, y_test),
                'rf': rf_model.score(X_test, y_test),
                'xgb': xgb_model.score(X_test, y_test)
            },
            'feature_names': X.columns.tolist()
        }
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None

# Sidebar for navigation
with st.sidebar:
    st.header("🔧 Navigation")
    
    page = st.radio(
        "Go to:",
        ["📊 Prediction", "📈 Model Analysis", "📁 Data Overview"]
    )
    
    st.header("📝 Student Information")
    
    # Input fields with realistic defaults
    col1, col2 = st.columns(2)
    
    with col1:
        jenis_kelamin = st.selectbox(
            "Gender",
            ["Laki-laki", "Perempuan"],
            index=0
        )
        
        umur = st.slider(
            "Age",
            min_value=18,
            max_value=25,
            value=20,
            step=1
        )
        
        status_menikah = st.selectbox(
            "Marital Status",
            ["Belum Menikah", "Menikah"],
            index=0
        )
        
        kehadiran = st.slider(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=75,
            step=1
        )
        
    with col2:
        partisipasi_diskusi = st.slider(
            "Discussion Participation (%)",
            min_value=0,
            max_value=100,
            value=70,
            step=1
        )
        
        nilai_tugas = st.slider(
            "Assignment Score (%)",
            min_value=0,
            max_value=100,
            value=75,
            step=1
        )
        
        aktivitas_elearning = st.slider(
            "E-learning Activity (%)",
            min_value=0,
            max_value=100,
            value=70,
            step=1
        )
        
        ipk = st.slider(
            "GPA",
            min_value=1.5,
            max_value=4.0,
            value=3.0,
            step=0.1
        )
    
    predict_button = st.button(
        "🔮 Predict Graduation Status",
        type="primary",
        use_container_width=True
    )

# Main content based on selected page
if page == "📊 Prediction":
    st.header("Model Predictions")
    
    # Train or load models
    if not st.session_state.models_trained:
        with st.spinner("Training models... This may take a few seconds"):
            models_data = train_models()
            if models_data:
                st.session_state.models = models_data
                st.session_state.models_trained = True
                st.success("✅ Models trained successfully!")
    
    if predict_button and st.session_state.models_trained:
        # Prepare input data
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
        
        # Get models
        dt_model = st.session_state.models['dt_model']
        rf_model = st.session_state.models['rf_model']
        xgb_model = st.session_state.models['xgb_model']
        
        # Make predictions
        dt_pred = dt_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]
        xgb_pred = xgb_model.predict(input_data)[0]
        
        # Get probabilities
        dt_proba = dt_model.predict_proba(input_data)[0]
        rf_proba = rf_model.predict_proba(input_data)[0]
        xgb_proba = xgb_model.predict_proba(input_data)[0]
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Decision Tree",
                value="LULUS" if dt_pred == 1 else "TIDAK LULUS",
                delta=f"{max(dt_proba)*100:.1f}% confidence"
            )
        
        with col2:
            st.metric(
                label="Random Forest",
                value="LULUS" if rf_pred == 1 else "TIDAK LULUS",
                delta=f"{max(rf_proba)*100:.1f}% confidence"
            )
        
        with col3:
            st.metric(
                label="XGBoost",
                value="LULUS" if xgb_pred == 1 else "TIDAK LULUS",
                delta=f"{max(xgb_proba)*100:.1f}% confidence"
            )
        
        # Consensus prediction
        votes = [dt_pred, rf_pred, xgb_pred]
        consensus = 'LULUS' if sum(votes) >= 2 else 'TIDAK LULUS'
        
        st.markdown("---")
        
        if consensus == 'LULUS':
            st.success(f"""
            ### 🎉 Consensus Prediction: **{consensus}**
            
            **Majority Vote:** {votes.count(1)} of 3 models predict graduation
            """)
        else:
            st.error(f"""
            ### ⚠️ Consensus Prediction: **{consensus}**
            
            **Majority Vote:** {votes.count(0)} of 3 models predict not graduating
            """)
        
        # Detailed probabilities
        with st.expander("📊 View Detailed Probabilities"):
            prob_df = pd.DataFrame({
                'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
                'Probability of Not Graduating': [dt_proba[0], rf_proba[0], xgb_proba[0]],
                'Probability of Graduating': [dt_proba[1], rf_proba[1], xgb_proba[1]]
            })
            st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)
    
    elif not st.session_state.models_trained:
        st.warning("⚠️ Models are still training. Please wait...")

elif page == "📈 Model Analysis":
    st.header("Model Performance Analysis")
    
    if st.session_state.models_trained:
        # Show accuracy metrics
        st.subheader("Model Accuracy Scores")
        
        acc_data = st.session_state.models['accuracy']
        acc_df = pd.DataFrame({
            'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
            'Accuracy': [acc_data['dt'], acc_data['rf'], acc_data['xgb']]
        })
        
        # Display as metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Decision Tree", f"{acc_data['dt']*100:.1f}%")
        col2.metric("Random Forest", f"{acc_data['rf']*100:.1f}%")
        col3.metric("XGBoost", f"{acc_data['xgb']*100:.1f}%")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        # Get feature importance from Random Forest (most reliable)
        rf_model = st.session_state.models['rf_model']
        feature_names = st.session_state.models['feature_names']
        importances = rf_model.feature_importances_
        
        # Create feature importance dataframe
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Display as bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi_df['Feature'], fi_df['Importance'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Random Forest Feature Importance')
        st.pyplot(fig)
        
        # Display as table
        with st.expander("View Feature Importance Table"):
            st.dataframe(fi_df, use_container_width=True)
    
    else:
        st.info("👈 Go to Prediction page first to train the models")

elif page == "📁 Data Overview":
    st.header("Dataset Overview")
    
    try:
        # Load and display data
        df = pd.read_csv('csv_Dataset_Mahasiswa_Kehadiran_Aktivitas_IPK.csv')
        
        # Basic statistics
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Students", len(df))
        
        with col2:
            lulus_count = df[df['status_akademik'] == 'Lulus'].shape[0]
            st.metric("Graduated", f"{lulus_count} ({lulus_count/len(df)*100:.1f}%)")
        
        with col3:
            tidak_count = df[df['status_akademik'] == 'Tidak'].shape[0]
            st.metric("Not Graduated", f"{tidak_count} ({tidak_count/len(df)*100:.1f}%)")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Distribution charts
        st.subheader("Data Distribution")
        
        tab1, tab2, tab3 = st.tabs(["GPA Distribution", "Attendance", "Graduation Status"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df['ipk'], bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('GPA')
            ax.set_ylabel('Frequency')
            ax.set_title('GPA Distribution')
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df['kehadiran'], bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax.set_xlabel('Attendance (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Attendance Distribution')
            st.pyplot(fig)
        
        with tab3:
            fig, ax = plt.subplots(figsize=(6, 4))
            status_counts = df['status_akademik'].value_counts()
            ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title('Graduation Status Distribution')
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>🎓 Student Graduation Prediction System | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
