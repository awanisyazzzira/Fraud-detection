<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>

<h1>Fraud Detection with Machine Learning</h1>


<h2>Data Exploration and Fraud Detection</h2>

<p>This project is focused on detecting credit card fraud using machine learning techniques. The dataset, retrieved from <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Kaggle</a>, contains 0.172% frauds.</p>


<h2>Exploring the Data</h2>

<p>We start by loading the credit card transaction dataset and performing exploratory data analysis (EDA). This includes calculating the correlation matrix and creating a heatmap to visualize the correlations between features.</p>

<pre><code># Calculate the correlation matrix
correlation_matrix = data.corr()
# Create a heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
</code></pre>

<h2>Data Preprocessing</h2>

<p>Next, we preprocess the data by handling class imbalance. We use the Synthetic Minority Over-sampling Technique (SMOTE) to oversample the minority class (frauds), creating a more balanced dataset for model training.</p>

<pre><code># Handling imbalance data by oversampling minority data (frauds)
smote = SMOTE(random_state=42) 
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
</code></pre>

<h2>Model Building</h2>

<p>We build a machine learning model, specifically a Random Forest Classifier, to detect fraudulent transactions. The model's hyperparameters are fine-tuned using GridSearchCV for optimal performance.</p>

<pre><code># Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [15, 20],
    'max_depth': [30, 40],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, scoring='f1')
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_
</code></pre>

<h2>Model Evaluation</h2>

<p>We evaluate the model's performance using various metrics, including the confusion matrix and the Area Under the Precision-Recall Curve (AUPRC). These metrics help us understand how well the model is identifying fraudulent transactions.</p>

<pre><code># Model testing
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
</code></pre>

<h2>Contributing</h2>

<p>If you'd like to contribute to this project, feel free to submit a pull request.</p>

<h2>License</h2>

<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
