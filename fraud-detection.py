import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,  
)
from imblearn.over_sampling import SMOTE
from collections import Counter

# Data retrieval
data = pd.read_csv(r'C:\Users\User\Desktop\fraud-detection\creditcard.csv')
data.head(10)

# EDA - Do this first before machine learning building and analysis
# Calculate the correlation matrix
correlation_matrix = data.corr()
# Create a heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Assigning features(X) and response variable(y)
X = data.drop(columns=['Class']) 
y = data['Class'] 
# Split the dataset into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Handling imbalance data by oversampling minority data (frauds)
smote = SMOTE(random_state=42) 
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", dict(Counter(y_train)))
print("After SMOTE:", dict(Counter(y_train_resampled)))

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [15, 20],
    'max_depth': [30, 40],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, scoring='f1')
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_

# Model testing
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 16})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model's performance using a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid(True)
plt.show()

# Calculate Area Under Precision-Recall Curve
auprc = average_precision_score(y_test, y_pred)
print(f'AUPRC: {auprc:.4f}')