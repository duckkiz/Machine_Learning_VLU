# -*- coding: utf-8 -*-
"""Chọn ngưỡng trên tập học để đạt mức recall tối thiểu."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/sinh_vien_mo_rong.csv")
X = df.drop(columns="qua_mon")
y = df["qua_mon"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
lr = make_pipeline(StandardScaler(), LogisticRegression())

xac_suat_cv = cross_val_predict(
    lr, X_tr, y_tr, cv=5, method="predict_proba"
)[:, 1]
luoi_nguong = np.round(np.arange(0.01, 1.00, 0.01), 2)
ket_qua_cv = []
for nguong in luoi_nguong:
    du_doan_cv = (xac_suat_cv >= nguong).astype(int)
    ket_qua_cv.append(
        (
            nguong,
            recall_score(y_tr, du_doan_cv, zero_division=0),
            precision_score(y_tr, du_doan_cv, zero_division=0),
            int(du_doan_cv.sum()),
        )
    )

nguong_hop_le = [dong for dong in ket_qua_cv if dong[1] >= 0.8]
if not nguong_hop_le:
    raise RuntimeError("Khong tim thay nguong dat recall CV toi thieu 0.8.")
nguong_tot = max(dong[0] for dong in nguong_hop_le)
print("Cac nguong quanh dap an (du lieu CV tren tap hoc)")
print(f"{'Nguong':>8} {'Recall CV':>12} {'Precision CV':>14} {'So bi bao':>12}")
for dong in ket_qua_cv:
    if abs(dong[0] - nguong_tot) <= 0.02 + 1e-9:
        print(f"{dong[0]:>8.2f} {dong[1]:>12.4f} {dong[2]:>14.4f} {dong[3]:>12}")
print(f"Nguong duoc chon tu tap hoc: {nguong_tot:.2f}")

# Chỉ đánh giá tập kiểm tra một lần sau khi đã chọn xong ngưỡng.
lr.fit(X_tr, y_tr)
xac_suat_te = lr.predict_proba(X_te)[:, 1]
du_doan_te = (xac_suat_te >= nguong_tot).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, du_doan_te, labels=[0, 1]).ravel()
recall_te = recall_score(y_te, du_doan_te, zero_division=0)
precision_te = precision_score(y_te, du_doan_te, zero_division=0)
print("Danh gia tap kiem tra (mot lan)")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Recall={recall_te:.4f}")
print(f"Precision={precision_te:.4f}")
print(f"So ban bi bao (TP+FP)={tp + fp}")

"""
Ngưỡng được chọn chỉ từ dự đoán ngoài mẫu trên tập học, không dùng kết quả tập
kiểm tra để tinh chỉnh. Recall kiểm tra có thể khác 0.8 vì số người thuộc lớp
dương hữu hạn, và mỗi mẫu dương làm recall thay đổi theo một bước rời rạc. Precision
và số người bị báo thể hiện chi phí cảnh báo khi hạ ngưỡng để tìm thêm lớp dương.
Nếu điều chỉnh ngưỡng sau khi xem tập kiểm tra, tập này không còn là phép đánh giá
độc lập và kết quả sẽ thiên lệch lạc quan.
"""
print("Ket luan: nguong duoc chon bang CV; khong dieu chinh theo tap kiem tra.")
