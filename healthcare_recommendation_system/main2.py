import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from transformers import pipeline
import streamlit as st

# Load the dataset
df = pd.read_csv(r'C:\Users\tanis\OneDrive\Documents\Tanishi\internship\Tekrosta Cloud\Disease_symptom_and_patient_profile_dataset.csv')

# DATA PREPROCESSING

# Convert 'Positive' to 1 and 'Negative' to 0 in the 'Outcome Variable' target column
df['Outcome Variable'] = df['Outcome Variable'].map({'Positive': 1, 'Negative': 0})

# Separate numerical and categorical columns, excluding 'Disease' from numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col != 'Outcome Variable']  # Ensure 'Outcome Variable' is excluded
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Ensure necessary columns are present before dropping
required_cols = ['Age', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Disease']
if not set(required_cols).issubset(df.columns):
    raise ValueError("Some required columns are missing.")

# Fill missing values for numerical data
numerical_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_cols)

# Set 'Outcome Variable' as target column
target_col = 'Outcome Variable'
y = df.pop(target_col)

# FEATURE ENGINEERING

# Identify the one-hot encoded symptom columns
symptom_cols = [col for col in df.columns if col.startswith(('Fever', 'Cough', 'Fatigue', 'Difficulty Breathing'))]

# Create a new feature 'Symptom Severity Score' as a sum of all symptom binary indicators
df['Symptom Severity Score'] = df[symptom_cols].sum(axis=1)

# Create interaction terms (e.g., Age with Symptom Severity Score)
df['Age_Symptom_Interaction'] = df['Age'] * df['Symptom Severity Score']

# Normalize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# MODEL DEVELOPMENT

# Remove classes with only one instance
class_counts = y.value_counts()
to_remove = class_counts[class_counts == 1].index
mask = ~y.isin(to_remove)
X = df[mask]
y = y[mask]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Neural Network': MLPClassifier(random_state=42, max_iter=300)
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], average='weighted')

    return accuracy, precision, recall, f1, auc, classification_report(y_test, y_pred)

# Evaluate all models
results = {}
for model_name, model in models.items():
    accuracy, precision, recall, f1, auc, report = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc,
        'Classification Report': report
    }

# RECOMMENDATION SYSTEM

required_cols = ['Age', 'Symptom Severity Score', 'Age_Symptom_Interaction']
if not set(required_cols).issubset(df.columns):
    raise ValueError("Some required columns are missing.")

classification_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text_generation_pipeline = pipeline("text-generation", model="gpt2")

# Function to generate recovery steps based on disease
def generate_recovery_steps(disease):
    prompt = disease
    response = text_generation_pipeline(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

def generate_recommendations(user_input):
    try:
        symptoms = [symptom for symptom in user_input.split() if symptom.lower() in ['fever', 'cough', 'fatigue', 'difficulty breathing']]
        if not symptoms:
            return "No relevant symptoms found in the input."

        candidate_labels = df.columns[df.columns.str.startswith('Disease_')].tolist()

        classification = classification_pipeline(user_input, candidate_labels, multi_label=True)

        predictions = classification['labels']
        confidence_scores = classification['scores']

        recommendations = sorted(zip(predictions, confidence_scores), key=lambda x: x[1], reverse=True)

        result = "Here's what we suggest:\n\n"
        for disease, score in recommendations:
            disease_name = disease.replace('Disease_', '')
            recovery_steps = generate_recovery_steps(disease_name)
            result += f"{disease_name}: (Confidence: {score:.2f})\n"
            result += f"Recovery Steps: {recovery_steps}\n\n"
        
        result += "Hope you found this helpful!"
        return result
    
    except Exception as e:
        # Handle errors
        st.write(f"An error occurred: {e}")
        return None


st.write("# Hi, welcome to our Healthcare Recommendation System")

# User input for symptoms
user_input = st.text_input("Enter symptoms here (like fever, cough, fatigue, etc):")

if user_input:
    recommendation = generate_recommendations(user_input)

    if recommendation:
        st.write("### Recommendations:")
        st.write(recommendation)
    else:
        st.write("Failed to generate recommendation.")
