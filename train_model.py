import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Load training dataset
df = pd.read_csv("student_performance_large.csv")

X = df[["math score", "reading score", "writing score"]]
y = (df[["math score", "reading score", "writing score"]].mean(axis=1) > 50).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Calculate and save accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

with open("accuracy.pkl", "wb") as f:
    pickle.dump(round(accuracy, 2), f)

print(f"Model trained with accuracy: {accuracy:.2f}%")
