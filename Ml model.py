import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🔹 Step 1: Load Dataset (Replace 'your_dataset.csv' with actual filename)
data = pd.read_csv("C:\\Users\\haroo\\TETREX\\venv\\data_file.csv")

print(f"✅ Dataset Loaded! Shape: {data.shape}")  # Shows (rows, columns)
print(data.dtypes)

# Remove text columns before training
data = data.drop(columns=["FileName", "md5Hash"])

# 🔹 Step 2: Preprocess Data
# Assuming the last column is the target ('Ransomware?') and the rest are features
X = data.iloc[:, :-1]  # All columns except last (features)
y = data.iloc[:, -1]   # Last column (target: 1 = ransomware, 0 = safe)

# 🔹 Step 3: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 4: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Step 5: Evaluate Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 🔹 Step 6: Save Model for Later Use
joblib.dump(model, "ransomware_model.pkl")
print("✅ Model saved as 'ransomware_model.pkl'")
