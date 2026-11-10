# -*- coding: utf-8 -*-
"""Vẽ đường ROC và so sánh khả năng phân biệt hai mô hình."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/sinh_vien_mo_rong.csv")
X = df.drop(columns="qua_mon")
y = df["qua_mon"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Rung": RandomForestClassifier(200, random_state=0),
}

duong_roc = {}
for ten, bo_phan_loai in mo_hinh.items():
    bo_phan_loai.fit(X_tr, y_tr)
    xac_suat = bo_phan_loai.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, xac_suat)
    auc = roc_auc_score(y_te, xac_suat)
    duong_roc[ten] = (fpr, tpr, auc)

plt.figure(figsize=(8, 6))
for ten, (fpr, tpr, auc) in duong_roc.items():
    plt.plot(fpr, tpr, label=f"{ten} (AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Ngau nhien")
plt.xlabel("Ty le bao dong nham")
plt.ylabel("Recall")
plt.title("Duong ROC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("baitap07/roc.png")
plt.close()

print(f"AUC Logistic: {duong_roc['Logistic'][2]:.4f}")
print(f"AUC Rung: {duong_roc['Rung'][2]:.4f}")

luoi_fpr = np.linspace(0.0, 1.0, 200)
tpr_logistic = np.interp(
    luoi_fpr, duong_roc["Logistic"][0], duong_roc["Logistic"][1]
)
tpr_rung = np.interp(luoi_fpr, duong_roc["Rung"][0], duong_roc["Rung"][1])
hieu = tpr_logistic - tpr_rung
cac_khoang_cat = []
dau_hieu_khac_khong = np.sign(hieu)
chi_so_khac_khong = np.flatnonzero(dau_hieu_khac_khong)
for truoc, sau in zip(chi_so_khac_khong[:-1], chi_so_khac_khong[1:]):
    if dau_hieu_khac_khong[truoc] != dau_hieu_khac_khong[sau]:
        khoang = (luoi_fpr[truoc], luoi_fpr[sau])
        if cac_khoang_cat and khoang[0] <= cac_khoang_cat[-1][1]:
            cac_khoang_cat[-1] = (
                cac_khoang_cat[-1][0],
                max(cac_khoang_cat[-1][1], khoang[1]),
            )
        else:
            cac_khoang_cat.append(khoang)
if cac_khoang_cat:
    print("Cac khoang FPR co giao cat (noi suy tren luoi):")
    for dau, cuoi in cac_khoang_cat:
        print(f"[{dau:.3f}, {cuoi:.3f}]")
else:
    print("Khong thay khoang giao cat noi bo tren luoi FPR noi suy.")

auc_logistic = duong_roc["Logistic"][2]
auc_rung = duong_roc["Rung"][2]
ten_tot_hon = "Logistic" if auc_logistic >= auc_rung else "Rung"

"""
Đường ROC càng áp sát góc trên bên trái thì mô hình càng đạt recall cao với tỷ
lệ báo động nhầm thấp. Hai đường có thể đổi thứ tự theo từng vùng FPR; các khoảng
giao cắt được dò bằng cách nội suy TPR trên cùng một lưới. AUC lớn hơn biểu thị
khả năng xếp ngẫu nhiên một mẫu lớp dương cao hơn một mẫu lớp âm tốt hơn. Kết quả
này chỉ so sánh mô hình trên tập kiểm tra của bộ dữ liệu thay thế qua_mon.
"""
print(f"Ket luan: {ten_tot_hon} co AUC lon hon tren tap kiem tra.")
