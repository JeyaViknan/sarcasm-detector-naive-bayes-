import streamlit as st
import pandas as pd
import numpy as np
import json
import string
import re
import joblib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sarcasm Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_dataset(file_path):
    """Load JSON lines dataset into a pandas DataFrame."""
    def parse_data(file):
        for line in open(file, 'r'):
            yield json.loads(line)
    data = list(parse_data(file_path))
    return pd.DataFrame(data)

def clean_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_and_prepare_data():
    """Load and prepare data with caching."""
    df = load_dataset("Sarcasm_Headlines_Dataset.json")
    df['cleaned'] = df['headline'].apply(clean_text)
    return df

@st.cache_resource
def train_all_models(_X_train, _X_test, _y_train, _y_test):
    """Train all models with caching."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(_X_train, _y_train)
        y_pred = model.predict(_X_test)
        
        results[name] = {
            'Accuracy': accuracy_score(_y_test, y_pred),
            'Precision': precision_score(_y_test, y_pred),
            'Recall': recall_score(_y_test, y_pred),
            'F1-Score': f1_score(_y_test, y_pred),
            'Model': model
        }
        predictions[name] = y_pred
    
    return results, predictions

def predict_sarcasm(sentence, model, tfidf):
    """Predict if a sentence is sarcastic."""
    cleaned = clean_text(sentence)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.decision_function(vectorized)[0] if hasattr(model, 'decision_function') else None
    return prediction, confidence

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<div class="header-title">🎭 Sarcasm Detection System</div>', unsafe_allow_html=True)
    st.markdown("An industry-ready machine learning system for detecting sarcasm in text headlines.")
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select a section:",
        ["🏠 Overview", "📈 Data Analysis", "🤖 Model Comparison", "🔮 Make Predictions", "📋 Dataset"]
    )
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_and_prepare_data()
    
    # ============================================================
    # PAGE: OVERVIEW
    # ============================================================
    if page == "🏠 Overview":
        st.header("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Headlines", len(df), delta=None)
        with col2:
            sarcastic_count = (df['is_sarcastic'] == 1).sum()
            st.metric("Sarcastic", sarcastic_count, delta=f"{sarcastic_count/len(df)*100:.1f}%")
        with col3:
            non_sarcastic_count = (df['is_sarcastic'] == 0).sum()
            st.metric("Not Sarcastic", non_sarcastic_count, delta=f"{non_sarcastic_count/len(df)*100:.1f}%")
        with col4:
            st.metric("Features (TF-IDF)", "5000", delta=None)
        
        st.divider()
        
        st.subheader("📋 Sample Headlines")
        sample_df = df[['headline', 'is_sarcastic']].head(10).copy()
        sample_df['Type'] = sample_df['is_sarcastic'].map({0: '❌ Not Sarcastic', 1: '✅ Sarcastic'})
        st.dataframe(sample_df[['headline', 'Type']], use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("🎯 System Pipeline")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.info("📥\n**Data Loading**\nJSON Dataset")
        with col2:
            st.info("🧹\n**Cleaning**\nText Preprocessing")
        with col3:
            st.info("🔢\n**Features**\nTF-IDF Vectorization")
        with col4:
            st.info("🤖\n**Training**\n4 ML Models")
        with col5:
            st.info("🎯\n**Prediction**\nReal-time Detection")
    
    # ============================================================
    # PAGE: DATA ANALYSIS
    # ============================================================
    elif page == "📈 Data Analysis":
        st.header("Exploratory Data Analysis")
        
        # Class Distribution
        st.subheader("📊 Class Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            counts = df['is_sarcastic'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#3498db', '#e74c3c']
            ax.bar(['Not Sarcastic', 'Sarcastic'], counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            percentages = df['is_sarcastic'].value_counts(normalize=True) * 100
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(percentages.values, labels=['Not Sarcastic', 'Sarcastic'], autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax.set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
            st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        # Word Clouds
        st.subheader("☁️ Word Clouds")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Not Sarcastic Headlines**")
            non_sarcastic_text = ' '.join(df[df['is_sarcastic'] == 0]['cleaned'].values)
            wc_non = WordCloud(width=400, height=300, background_color='white', colormap='Blues').generate(non_sarcastic_text)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wc_non, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.write("**Sarcastic Headlines**")
            sarcastic_text = ' '.join(df[df['is_sarcastic'] == 1]['cleaned'].values)
            wc_sarcastic = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(sarcastic_text)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wc_sarcastic, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        # Top Words
        st.subheader("📝 Top 20 Most Frequent Words")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Not Sarcastic**")
            non_sarcastic_words = ' '.join(df[df['is_sarcastic'] == 0]['cleaned'].values).split()
            non_sarcastic_freq = Counter(non_sarcastic_words).most_common(20)
            words_ns, counts_ns = zip(*non_sarcastic_freq)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(words_ns)), counts_ns, color='#3498db', alpha=0.8, edgecolor='black')
            ax.set_yticks(range(len(words_ns)))
            ax.set_yticklabels(words_ns)
            ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.write("**Sarcastic**")
            sarcastic_words = ' '.join(df[df['is_sarcastic'] == 1]['cleaned'].values).split()
            sarcastic_freq = Counter(sarcastic_words).most_common(20)
            words_s, counts_s = zip(*sarcastic_freq)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(words_s)), counts_s, color='#e74c3c', alpha=0.8, edgecolor='black')
            ax.set_yticks(range(len(words_s)))
            ax.set_yticklabels(words_s)
            ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig, use_container_width=True)
    
    # ============================================================
    # PAGE: MODEL COMPARISON
    # ============================================================
    elif page == "🤖 Model Comparison":
        st.header("Machine Learning Models Comparison")
        
        with st.spinner("Training models..."):
            # Feature extraction
            tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            X = tfidf.fit_transform(df['cleaned'])
            y = df['is_sarcastic']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models
            results, predictions = train_all_models(X_train, X_test, y_train, y_test)
        
        # Model Comparison Table
        st.subheader("📊 Performance Metrics")
        df_results = pd.DataFrame({
            model: {metric: value for metric, value in metrics.items() if metric != 'Model'}
            for model, metrics in results.items()
        }).T
        
        st.dataframe(df_results.round(4), use_container_width=True)
        
        st.divider()
        
        # Model Comparison Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Accuracy Comparison**")
            fig, ax = plt.subplots(figsize=(8, 5))
            df_results['Accuracy'].plot(kind='bar', ax=ax, color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax.set_ylim([0.5, 1.0])
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.write("**F1-Score Comparison**")
            fig, ax = plt.subplots(figsize=(8, 5))
            df_results['F1-Score'].plot(kind='bar', ax=ax, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
            ax.set_ylim([0.5, 1.0])
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        # Confusion Matrices
        st.subheader("🎯 Confusion Matrices")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (model_name, y_pred) in enumerate(predictions.items()):
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Not Sarcastic', 'Sarcastic'],
                        yticklabels=['Not Sarcastic', 'Sarcastic'],
                        cbar_kws={'label': 'Count'})
            axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        # Best Model
        best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
        best_model = results[best_model_name]['Model']
        best_metrics = results[best_model_name]
        
        st.subheader("🏆 Best Model")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", best_model_name)
        with col2:
            st.metric("Accuracy", f"{best_metrics['Accuracy']:.4f}")
        with col3:
            st.metric("Precision", f"{best_metrics['Precision']:.4f}")
        with col4:
            st.metric("F1-Score", f"{best_metrics['F1-Score']:.4f}")
        
        # Store in session state for predictions
        st.session_state.best_model = best_model
        st.session_state.tfidf = tfidf
    
    # ============================================================
    # PAGE: PREDICTIONS
    # ============================================================
    elif page == "🔮 Make Predictions":
        st.header("Real-Time Sarcasm Detection")
        
        # Check if models are trained
        if 'best_model' not in st.session_state or 'tfidf' not in st.session_state:
            st.info("⚠️ Please visit the 'Model Comparison' section first to train the models.")
        else:
            best_model = st.session_state.best_model
            tfidf = st.session_state.tfidf
            
            st.subheader("✍️ Enter a Headline")
            user_input = st.text_area("Type or paste a headline:", height=100, placeholder="e.g., 'Oh great, another Monday morning!'")
            
            if st.button("🔍 Detect Sarcasm", use_container_width=True, type="primary"):
                if user_input.strip():
                    prediction, confidence = predict_sarcasm(user_input, best_model, tfidf)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if prediction == 1:
                            st.success("✅ **SARCASTIC**", icon="🎭")
                            st.write("This headline appears to be **sarcastic**.")
                        else:
                            st.info("❌ **NOT SARCASTIC**", icon="📰")
                            st.write("This headline appears to be **not sarcastic**.")
                    
                    with col2:
                        if confidence is not None:
                            confidence_pct = abs(confidence) * 100
                            st.metric("Confidence", f"{min(confidence_pct, 100):.1f}%")
                else:
                    st.warning("Please enter a headline to analyze.")
            
            st.divider()
            
            st.subheader("📚 Try Sample Headlines")
            samples = [
                "Oh great, another Monday morning!",
                "The weather is beautiful today",
                "I just love waiting in traffic for hours",
                "Scientists discover new species in the Amazon",
                "Sure, let me just work for free, that sounds amazing"
            ]
            
            for i, sample in enumerate(samples):
                if st.button(f"📌 {sample}", key=f"sample_{i}"):
                    prediction, confidence = predict_sarcasm(sample, best_model, tfidf)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if prediction == 1:
                            st.success("✅ **SARCASTIC**")
                        else:
                            st.info("❌ **NOT SARCASTIC**")
                    with col2:
                        if confidence is not None:
                            st.metric("Confidence", f"{min(abs(confidence)*100, 100):.1f}%")
    
    # ============================================================
    # PAGE: DATASET
    # ============================================================
    elif page == "📋 Dataset":
        st.header("Dataset Explorer")
        
        st.subheader("📊 Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Sarcastic", (df['is_sarcastic'] == 1).sum())
        with col3:
            st.metric("Not Sarcastic", (df['is_sarcastic'] == 0).sum())
        
        st.divider()
        
        st.subheader("🔍 Browse Headlines")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox("Filter by type:", ["All", "Sarcastic", "Not Sarcastic"])
        with col2:
            num_rows = st.slider("Number of rows to display:", 5, 100, 20)
        
        # Apply filter
        if filter_type == "Sarcastic":
            filtered_df = df[df['is_sarcastic'] == 1]
        elif filter_type == "Not Sarcastic":
            filtered_df = df[df['is_sarcastic'] == 0]
        else:
            filtered_df = df
        
        # Display table
        display_df = filtered_df[['headline', 'is_sarcastic']].head(num_rows).copy()
        display_df['Type'] = display_df['is_sarcastic'].map({0: '❌ Not Sarcastic', 1: '✅ Sarcastic'})
        st.dataframe(display_df[['headline', 'Type']], use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("📥 Download Dataset")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="sarcasm_dataset.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
