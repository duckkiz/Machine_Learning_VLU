# -*- coding: utf-8 -*-
"""So sánh hàm đánh giá tự viết với các chỉ số của scikit-learn."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia(y_that, y_doan):
    """Tính accuracy, precision, recall và F1 cho lớp dương 1 bằng numpy."""
    y_that = np.asarray(y_that).ravel()
    y_doan = np.asarray(y_doan).ravel()
    tp = np.sum((y_that == 1) & (y_doan == 1))
    tn = np.sum((y_that == 0) & (y_doan == 0))
    fp = np.sum((y_that == 0) & (y_doan == 1))
    fn = np.sum((y_that == 1) & (y_doan == 0))

    def chia(tu_so, mau_so, ten_chi_so):
        if mau_so == 0:
            print(
                f"Canh bao: chi so {ten_chi_so} khong xac dinh "
                "do mau so bang 0; tra ve 0.0."
            )
            # Cach xu ly nay tuong duong zero_division=0 cua sklearn.
            return 0.0
        return tu_so / mau_so

    accuracy = chia(tp + tn, len(y_that), "accuracy")
    precision = chia(tp, tp + fp, "precision")
    recall = chia(tp, tp + fn, "recall")
    f1 = chia(2 * tp, 2 * tp + fp + fn, "F1")
    return accuracy, precision, recall, f1


df = pd.read_csv("data/sinh_vien_mo_rong.csv")
X = df.drop(columns="qua_mon")
y = df["qua_mon"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = {
    "Mo hinh luoi": DummyClassifier(strategy="most_frequent"),
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
}

print("So sanh cac chi so tren tap kiem tra (lop duong: qua_mon=1)")
print(
    f"{'Mo hinh':<16} {'Chi so':<12} {'Tu viet':>10} "
    f"{'sklearn':>10} {'khop':>8}"
)
for ten, bo_phan_loai in mo_hinh.items():
    bo_phan_loai.fit(X_tr, y_tr)
    du_doan = bo_phan_loai.predict(X_te)
    tu_viet = danh_gia(y_te, du_doan)
    sklearn = (
        accuracy_score(y_te, du_doan),
        precision_score(y_te, du_doan, zero_division=0),
        recall_score(y_te, du_doan, zero_division=0),
        f1_score(y_te, du_doan, zero_division=0),
    )
    for ten_chi_so, gia_tri_tu_viet, gia_tri_sklearn in zip(
        ("accuracy", "precision", "recall", "F1"), tu_viet, sklearn
    ):
        print(
            f"{ten:<16} {ten_chi_so:<12} {gia_tri_tu_viet:>10.4f} "
            f"{gia_tri_sklearn:>10.4f} "
            f"{str(bool(np.isclose(gia_tri_tu_viet, gia_tri_sklearn))):>8}"
        )

"""
Trong CSV thay thế, lớp qua môn (1) là lớp phổ biến nhất, nên mô hình lười đoán
dương cho mọi bạn. Precision vì vậy xác định được (TP+FP lớn hơn 0), còn recall
bằng 1; hàm chỉ in cảnh báo và trả 0.0 nếu một mẫu số thực sự bằng 0. Accuracy
của mô hình lười phản ánh tỷ lệ lớp phổ biến trong dữ liệu, không chứng minh rằng
mô hình phân biệt được hai lớp. Khác với bài toán bỏ học trong tài liệu, ở đây
không thể kết luận recall của mô hình lười bằng 0 vì nhãn và phân bố lớp đã đổi.
Nếu mô hình lười dự đoán toàn lớp âm như trường hợp bài bỏ học gốc, TP+FP sẽ
bằng 0 và precision không xác định; hàm sẽ cảnh báo và trả về 0.0.
"""
print("Ket luan: mo hinh luoi chi theo lop pho bien, khong phan biet duoc hai lop.")
