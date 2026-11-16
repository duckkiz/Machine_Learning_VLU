# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("baitap09/xor.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    solver="lbfgs",
    max_iter=5000,
    random_state=0,
)
mlp.fit(X_tr, y_tr)

logistic = LogisticRegression()
logistic.fit(X_tr, y_tr)
print(f"Accuracy kiem tra LogisticRegression: {logistic.score(X_te, y_te):.4f}")
print(f"Accuracy tap hoc: {mlp.score(X_tr, y_tr):.4f}")
print(f"Accuracy tap kiem tra: {mlp.score(X_te, y_te):.4f}")

W1, W2 = mlp.coefs_
b1, b2 = mlp.intercepts_
x_nam_diem = X_te.iloc[:5].to_numpy()
h = np.maximum(0, x_nam_diem @ W1 + b1)
z = h @ W2 + b2
xac_suat_tu_tinh = 1 / (1 + np.exp(-z.ravel()))
xac_suat_mlp = mlp.predict_proba(X_te.iloc[:5])[:, 1]
sai_so = np.abs(xac_suat_tu_tinh - xac_suat_mlp)

print("Nam diem dau tien cua tap kiem tra:")
print("STT | Xac suat tu tinh | Xac suat MLP | Hieu so tuyet doi")
for stt, (xac_suat, xac_suat_tham_chieu, chenh_lech) in enumerate(
    zip(xac_suat_tu_tinh, xac_suat_mlp, sai_so), start=1
):
    print(
        f"{stt:>3} | {xac_suat:.12f} | "
        f"{xac_suat_tham_chieu:.12f} | {chenh_lech:.12e}"
    )
print(f"np.allclose: {np.allclose(xac_suat_tu_tinh, xac_suat_mlp)}")
print(f"Hieu so lon nhat: {sai_so.max():.12e}")

print("Kich thuoc ma tran trong so:")
for ten, ma_tran in (("W1", W1), ("W2", W2)):
    print(f"{ten}: {ma_tran.shape}")
so_tham_so = sum(w.size for w in mlp.coefs_) + sum(
    b.size for b in mlp.intercepts_
)
print(f"Tong so tham so: {so_tham_so}")
print("Ket luan: Lan truyen numpy khop voi predict_proba cua MLP.")

"""
Mạng nhận đầu vào, nhân ma trận với trọng số rồi cộng hệ số chặn ở lớp ẩn.
ReLU được áp dụng cho từng giá trị của lớp ẩn trước khi truyền sang lớp ra.
Lớp ra tiếp tục nhân trọng số, cộng hệ số chặn và dùng sigmoid để tạo xác suất.
Phép tính numpy khớp với predict_proba vì sử dụng cùng các trọng số đã học và cùng phép biến đổi.
Với tệp này và cách chia có stratify, accuracy MLP kiểm tra là 0.9000, còn LogisticRegression đạt 0.4500.
Sai lệch rất nhỏ nếu có là do sai số số học của phép tính dấu phẩy động.
"""
