from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

csv_path = Path(__file__).resolve().parent / "sinh_vien.csv"
df = pd.read_csv(csv_path)

X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]

print("Bang F1 theo nguong:")

for t in np.arange(0.05, 1.0, 0.05):
    y_pred = (proba >= t).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"Ngưỡng: {t:.2f}, F1: {f1:.4f}")

best_t = max(np.arange(0.05, 1.0, 0.05), key=lambda t: f1_score(y_test, (proba >= t).astype(int), zero_division=0))
best_f1 = f1_score(y_test, (proba >= best_t).astype(int), zero_division=0)

print(f"\nNgunh tối ưu: {best_t:.2f}, F1 tối đa: {best_f1:.4f}")