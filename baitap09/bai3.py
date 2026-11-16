# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ket_qua = []
for act in ("logistic", "tanh", "relu"):
    mo_hinh = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100, 100),
            alpha=1.0,
            activation=act,
            solver="lbfgs",
            max_iter=20000,
            random_state=0,
        ),
    )
    mo_hinh.fit(X_tr, y_tr)
    diem_hoc = accuracy_score(y_tr, mo_hinh.predict(X_tr))
    diem_kiem_tra = accuracy_score(y_te, mo_hinh.predict(X_te))
    ket_qua.append((act, diem_hoc, diem_kiem_tra))

print("Ham kich hoat | Accuracy tap hoc | Accuracy tap kiem tra")
for act, diem_hoc, diem_kiem_tra in ket_qua:
    print(f"{act:<13} | {diem_hoc:.4f}           | {diem_kiem_tra:.4f}")

diem_cao_nhat = max(dong[2] for dong in ket_qua)
ham_dong_diem = [dong[0] for dong in ket_qua if dong[2] == diem_cao_nhat]
print(
    f"Ket luan: {', '.join(ham_dong_diem)} cung co accuracy kiem tra "
    f"cao nhat ({diem_cao_nhat:.4f})."
)

"""
Tanh và ReLU cùng đạt accuracy kiểm tra 0.9067, cao hơn logistic là 0.8533.
Accuracy tập học lần lượt là 0.8400, 0.8711 và 0.8844 cho logistic, tanh và ReLU.
Accuracy học của ReLU là 0.8844, làm tròn đến hai chữ số thập phân thành 0.88; accuracy kiểm tra khớp 0.9067.
Tập kiểm tra chỉ có 75 điểm nên mỗi điểm sai làm accuracy thay đổi khoảng 0.0133, do đó không nên diễn giải chênh lệch nhỏ quá mức.
"""
