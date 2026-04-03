import pandas as pd
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print('Loading data...')
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

X = df.drop('Attrition', axis=1)
y = df['Attrition']

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

print(f'Features: {len(num_cols)} numerical, {len(cat_cols)} categorical')

# Create pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipeline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    ))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Training model...')
pipeline_model.fit(X_train, y_train)

# Evaluate
y_pred = pipeline_model.predict(X_test)
y_proba = pipeline_model.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
    'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_test, y_proba))
}

print('\n=== Model Performance ===')
for k, v in metrics.items():
    print(f'{k.upper()}: {v:.4f}')

# Save confusion matrix
os.makedirs('static', exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('static/confusion.png', dpi=100, bbox_inches='tight')
plt.close()

# Save model and metrics
pickle.dump(pipeline_model, open('model.pkl', 'wb'))
pickle.dump(metrics, open('metrics.pkl', 'wb'))

print('\nFiles saved: model.pkl, metrics.pkl, static/confusion.png')
