<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>The Detective's Notebook: Uncovering Credit Card Fraud</h1>

![credit-card-fraud](path_to_image.png)

<h2>Project Overview</h2>

<p>In the world of finance, where digital transactions flow like a river, there's a silent crime - credit card fraud. Welcome to the detective's journey into data, where we deploy machine learning to unveil this hidden threat.</p>

<h2>The Case File</h2>

<p>Our investigation begins with a valuable dataset, sourced from <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Kaggle</a>. Within its vast landscape of 284,807 transactions, we're hunting for elusive prey - the 0.172% that's hiding in plain sight - the frauds.</p>

<h2>The Clues in the Data</h2>

<p>Before diving into the technical details, we perform exploratory data analysis (EDA) to understand the crime scene better. We employ Python's Pandas and Seaborn to build a correlation matrix, illuminating hidden connections between data points. Our heatmap visually guides us through this intricate web of relationships.</p>

<pre><code># Calculate the correlation matrix
correlation_matrix = data.corr()
# Create a heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
</code></pre>

<h2>Preparing for the Hunt</h2>

<p>Before we enter the battlefield, we need to prepare. Our primary challenge is data imbalance, a common trait of criminal investigations. Here, we employ SMOTE (Synthetic Minority Over-sampling Technique) to give life to the underrepresented minority class (frauds). This creates a balanced dataset, equipping us for the hunt.</p>

<pre><code># Handling imbalance data by oversampling minority data (frauds)
smote = SMOTE(random_state=42) 
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
</code></pre>

<h2>The Detective's Arsenal</h2>

<p>The detective's toolkit includes a Random Forest Classifier, a trusted companion in many battles. To ensure it's at its best, we use GridSearchCV to find the perfect hyperparameters.</p>

<pre><code># Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [15, 20],
    'max_depth': [30, 40],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, scoring='f1')
grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_
</code></pre>

<h2>Unmasking the Culprits</h2>

<p>With our toolkit ready, we enter the world of real-world data. The truth unravels through the eyes of a confusion matrix, revealing how our model predicts frauds. We evaluate its performance using metrics like precision, recall, and the subtle Area Under the Precision-Recall Curve (AUPRC).</p>

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

<h2>Join the Detective's Journey</h2>

<p>If you're curious about our detective's journey and wish to join the expedition, we welcome your contributions. Feel free to submit a pull request and become a fellow detective in this data mystery.</p>

