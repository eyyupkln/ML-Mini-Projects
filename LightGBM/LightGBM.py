import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

print("Loading Titanic dataset...\n")
df = sns.load_dataset('titanic')

y = df['survived']

X = df.drop(['survived', 'alive'], axis=1)


categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm_model = LGBMClassifier(random_state=42, n_jobs=-1)

print("LightGBM is building its asymmetrical trees. Watch the speed...")
start_time = time.time()

lgbm_model.fit(X_train, y_train)

end_time = time.time()
print(f"Training Complete! Time elapsed: {end_time - start_time:.4f} seconds.\n")

y_pred = lgbm_model.predict(X_test)

print("--- 🚀 FINAL TEST METRICS ---")
print(f"Accuracy: %{accuracy_score(y_test, y_pred) * 100:.2f}\n")

print("--- CONFUSION MATRIX ---")
cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred),
                     index=['Actual Dead (0)', 'Actual Survived (1)'],
                     columns=['Predicted Dead (0)', 'Predicted Survived (1)'])
print(cm_df)

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Dead (0)', 'Survived (1)']))



#Hyperparameter Tuning


from sklearn.model_selection import RandomizedSearchCV

lgb_model = LGBMClassifier(verbosity=-1)

param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
y_pred = random_search.predict(X_test)
print(f"Accuracy: %{accuracy_score(y_test, y_pred) * 100:.2f}\n")
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Dead (0)', 'Survived (1)']))
print(confusion_matrix(y_pred, y_test))

