# -*- coding: utf-8 -*-
"""So sánh ảnh hưởng của class_weight lên ba bộ phân loại."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


df = pd.read_csv("data/sinh_vien_mo_rong.csv")
X = df.drop(columns="qua_mon")
y = df["qua_mon"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cau_hinh = {
    "Logistic": (
        lambda: make_pipeline(
            StandardScaler(), LogisticRegression(class_weight=None)
        ),
        lambda: make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced")
        ),
    ),
    "Rung": (
        lambda: RandomForestClassifier(200, random_state=0),
        lambda: RandomForestClassifier(
            200, random_state=0, class_weight="balanced"
        ),
    ),
    "SVC": (
        lambda: make_pipeline(StandardScaler(), SVC(random_state=0)),
        lambda: make_pipeline(
            StandardScaler(), SVC(random_state=0, class_weight="balanced")
        ),
    ),
}

ket_qua = {}
for ten, (tao_truoc, tao_sau) in cau_hinh.items():
    ket_qua[ten] = {}
    for trang_thai, tao_mo_hinh in (
        ("Truoc", tao_truoc),
        ("Sau", tao_sau),
    ):
        mo_hinh = tao_mo_hinh()
        mo_hinh.fit(X_tr, y_tr)
        du_doan = mo_hinh.predict(X_te)
        ket_qua[ten][trang_thai] = {
            "precision": precision_score(y_te, du_doan, zero_division=0),
            "recall": recall_score(y_te, du_doan, zero_division=0),
            "F1": f1_score(y_te, du_doan, zero_division=0),
        }

print("So sanh truoc va sau khi can bang lop (lop duong: qua_mon=1)")
print(
    f"{'Mo hinh':<12} {'Trang thai':<12} {'Precision':>10} "
    f"{'Recall':>10} {'F1':>10} {'Delta P':>10} "
    f"{'Delta R':>10} {'Delta F1':>10}"
)
delta_lon_nhat_f1 = ("", -1.0)
delta_lon_nhat_recall = ("", -1.0)
for ten, cac_ket_qua in ket_qua.items():
    truoc = cac_ket_qua["Truoc"]
    sau = cac_ket_qua["Sau"]
    delta = {chi_so: sau[chi_so] - truoc[chi_so] for chi_so in truoc}
    for trang_thai in ("Truoc", "Sau"):
        gia_tri = cac_ket_qua[trang_thai]
        delta_hien_thi = delta if trang_thai == "Sau" else {
            chi_so: 0.0 for chi_so in delta
        }
        print(
            f"{ten:<12} {trang_thai:<12} {gia_tri['precision']:>10.4f} "
            f"{gia_tri['recall']:>10.4f} {gia_tri['F1']:>10.4f} "
            f"{delta_hien_thi['precision']:>+10.4f} "
            f"{delta_hien_thi['recall']:>+10.4f} "
            f"{delta_hien_thi['F1']:>+10.4f}"
        )
    if abs(delta["F1"]) > delta_lon_nhat_f1[1]:
        delta_lon_nhat_f1 = (ten, abs(delta["F1"]))
    if abs(delta["recall"]) > delta_lon_nhat_recall[1]:
        delta_lon_nhat_recall = (ten, abs(delta["recall"]))

print(
    f"Thay doi F1 lon nhat: {delta_lon_nhat_f1[0]} "
    f"(|delta|={delta_lon_nhat_f1[1]:.4f})"
)
print(
    f"Thay doi recall lon nhat: {delta_lon_nhat_recall[0]} "
    f"(|delta|={delta_lon_nhat_recall[1]:.4f})"
)

"""
Bảng cho thấy mức precision, recall và F1 trước/sau khi đặt class_weight=balanced
trên cùng một phép chia dữ liệu. Với dữ liệu hiện tại, cả ba mô hình cho cùng các
chỉ số trước và sau, nên mọi delta đều bằng 0 và không mô hình nào thay đổi nhiều
hơn mô hình khác. Đây là quan sát trên bộ dữ liệu nhỏ này, không chứng minh rằng
cân bằng lớp nói chung không ảnh hưởng. Dữ liệu dùng nhãn qua_mon nên các con số
không đối chiếu trực tiếp được với bài toán bỏ học.
"""
print(
    f"Ket luan: F1 doi nhieu nhat o {delta_lon_nhat_f1[0]}, "
    f"recall doi nhieu nhat o {delta_lon_nhat_recall[0]}."
)
