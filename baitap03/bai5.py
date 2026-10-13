import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]
print(f"Tỉ lệ lớp 'qua môn' = {y.mean():.3f}  (đoán bừa toàn lớp 1 đạt ~{y.mean():.3f})\n")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

with warnings.catch_warnings(record=True) as ds:
    warnings.simplefilter("always")
    mnb = MultinomialNB().fit(X_tr, y_tr)
    acc_m = accuracy_score(y_te, mnb.predict(X_te))
print(f"MultinomialNB : accuracy = {acc_m:.4f} | số cảnh báo/lỗi: {len(ds)}")
gnb = GaussianNB().fit(X_tr, y_tr)
acc_g = accuracy_score(y_te, gnb.predict(X_te))
print(f"GaussianNB    : accuracy = {acc_g:.4f}")

am, ag = [], []
for s in range(50):
    a, b, c, d = train_test_split(X, y, test_size=0.3, random_state=s, stratify=y)
    am.append(accuracy_score(d, MultinomialNB().fit(a, c).predict(b)))
    ag.append(accuracy_score(d, GaussianNB().fit(a, c).predict(b)))
print(f"\nTrung bình 50 lần chia: Multinomial = {np.mean(am):.4f}, Gaussian = {np.mean(ag):.4f}")

print("\nĐổi đơn vị đo (giờ -> phút, điểm thang 10 -> thang 100):")
for ten, hs in [("gốc", (1, 1)), ("gio_on x60 (phút)", (60, 1)), ("diem x10 (thang 100)", (1, 10)), ("cả hai", (60, 10))]:
    A, B = X_tr.values * hs, X_te.values * hs
    pm = MultinomialNB().fit(A, y_tr).predict(B)
    pg = GaussianNB().fit(A, y_tr).predict(B)
    print(f"  {ten:22s}: Multinomial = {accuracy_score(y_te, pm):.4f} | Gaussian = {accuracy_score(y_te, pg):.4f}")

Z = (X_tr - X_tr.mean()) / X_tr.std()
try:
    MultinomialNB().fit(Z, y_tr)
except ValueError as e:
    print(f"\nChuẩn hóa z-score rồi chạy MultinomialNB -> LỖI: {e}")
print(f"Cùng dữ liệu chuẩn hóa với GaussianNB -> accuracy = "
      f"{accuracy_score(y_te, GaussianNB().fit(Z, y_tr).predict((X_te - X_tr.mean()) / X_tr.std())):.4f}")

print("\nMultinomialNB học: xác suất 'từ' theo lớp (mỗi lớp là 1 phân phối trên 2 đặc trưng, tổng = 1)")
print(np.exp(mnb.feature_log_prob_).round(4), "  (cột: gio_on, diem_giua_ky)")
print("GaussianNB học: trung bình mỗi lớp (cột: gio_on, diem_giua_ky)")
print(gnb.theta_.round(3))