import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

file_path = '2. drug200.csv'
df = pd.read_csv(file_path)
print(df.head())

null_values = df.isna().sum()
print(null_values)

print(df['Drug'].value_counts())
print(df['Sex'].value_counts())

cross_tab_sex_drug = pd.crosstab(df['Sex'], df['Drug'])
sns.heatmap(cross_tab_sex_drug, annot=True, fmt='d', cmap='viridis', cbar=True)
plt.xlabel('Drug')
plt.ylabel('Sex')
plt.title('Cross-tabulation of Sex and Drug')
plt.savefig('cross_tab_sex_drug.png')  
plt.show()

cross_tab_drug_sex = pd.crosstab(df['Drug'], df['Sex'])
cross_tab_drug_sex.plot(kind='bar', color=['salmon', 'lightblue'])
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Distribution of Sex by Drug')
plt.savefig('cross_tab_drug_sex.png')  
plt.show()

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']

print(X.head())
y = df.iloc[:, -1].values
y = pd.DataFrame(y, columns=['Drug'])
print(y.head())

label_X = X.copy()
columns_to_encode = ['Sex', 'BP', 'Cholesterol']
label_encoder = LabelEncoder()

for col in columns_to_encode:
    label_X[col] = label_encoder.fit_transform(X[col])

print(label_X)

X_train, X_test, y_train, y_test = train_test_split(label_X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.values.ravel())

y_preds_rf = rf_model.predict(X_test)

accuracy_rf = rf_model.score(X_test, y_test)
print(f'\nAccuracy score for Random Forest is {accuracy_rf}')

conf_matrix_rf = confusion_matrix(y_test, y_preds_rf)
print(conf_matrix_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Random Forest Model')
plt.savefig('confusion_matrix_rf.png')  
plt.show()

class_report_rf = classification_report(y_test, y_preds_rf)
print(class_report_rf)

feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                  index=label_X.columns,
                                  columns=['Importance'])
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.index, y=feature_importances['Importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances in Random Forest Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importances_rf.png')  
plt.show()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(label_X, y.values.ravel())

best_params = grid_search.best_params_
print(best_params)

cross_val_scores_rf = cross_val_score(rf_model, label_X, y.values.ravel(), cv=5, scoring='accuracy')
print(cross_val_scores_rf)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), cross_val_scores_rf, marker='o', linestyle='-', color='b')
plt.xlabel('Fold')
plt.ylabel('Accuracy Score')
plt.title('Cross-Validation Scores for Random Forest Model')
plt.xticks(range(1, 6))
plt.grid(True)
plt.savefig('cross_val_scores_rf.png')  
plt.show()
