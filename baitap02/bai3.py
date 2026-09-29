from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

csv_path = Path(__file__).resolve().parent / "sinh_vien.csv"
df = pd.read_csv(csv_path)

X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print(f"Diem can bang (z = 0) tai gio_on = -b/w = {-b / w:.4f} gio")
print()


def du_doan(gio):
    """Nhan vao so gio on, in ra z, xac suat qua mon va nhan theo nguong 0.5."""
    z = w * gio + b                      # z = w*x + b
    p = 1 / (1 + np.exp(-z))             # sigmoid(z)
    nhan = 1 if p >= 0.5 else 0          # nguong 0.5
    print(f"gio = {gio:6.2f} | z = {z:8.4f} | P(qua mon) = {p:.4f} | nhan = {nhan}")
    return z, p, nhan


for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)

print()
print("Doi chieu voi predict_proba cua sklearn:")
for gio in [3, 8, 12.89, 18, 26]:
    p_lib = mo_hinh.predict_proba(pd.DataFrame({"gio_on": [gio]}))[0, 1]
    print(f"gio = {gio:6.2f} | P (sklearn) = {p_lib:.4f}")