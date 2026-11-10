# -*- coding: utf-8 -*-
"""Minh họa MAE, RMSE và R2 khi làm hỏng một dự đoán."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/gia_nha.csv")
print("Nam dong dau cua du lieu gia nha:")
print(df.head())
print("Kieu du lieu cac cot:")
print(df.dtypes)
print("Cot nhan: gia; don vi: ty dong (theo xac nhan cua nguoi dung).")

if not all(pd.api.types.is_numeric_dtype(df[cot]) for cot in df.columns):
    raise TypeError("Du lieu co cot khong phai so; can xac dinh cach xu ly truoc.")

X = df.drop(columns="gia")
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17
)
mo_hinh = LinearRegression()
mo_hinh.fit(X_tr, y_tr)
du_doan_truoc = mo_hinh.predict(X_te)
sai_so_truoc = np.asarray(y_te) - du_doan_truoc

du_doan_sau = du_doan_truoc.copy()
du_doan_sau[0] += 5.0
sai_so_sau = np.asarray(y_te) - du_doan_sau

def tinh_chi_so(sai_so, y_that):
    mae = np.mean(np.abs(sai_so))
    rmse = np.sqrt(np.mean(sai_so**2))
    r2 = r2_score(y_that, y_that - sai_so)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


chi_so_truoc = tinh_chi_so(sai_so_truoc, y_te)
chi_so_sau = tinh_chi_so(sai_so_sau, y_te)
thay_doi_phan_tram = {}
print("So sanh truoc va sau khi cong them 5 ty vao mot du doan")
print(
    f"{'Thuoc do':<10} {'Truoc':>12} {'Sau':>12} "
    f"{'Chenh lech':>14} {'Chenh lech (%)':>18}"
)
for ten in ("MAE", "RMSE", "R2"):
    truoc = chi_so_truoc[ten]
    sau = chi_so_sau[ten]
    chenh_lech = abs(sau - truoc)
    phan_tram = chenh_lech / abs(truoc) * 100 if truoc != 0 else float("inf")
    thay_doi_phan_tram[ten] = phan_tram
    print(
        f"{ten:<10} {truoc:>12.4f} {sau:>12.4f} "
        f"{chenh_lech:>14.4f} {phan_tram:>17.2f}%"
    )

thuoc_do_lon_nhat = max(thay_doi_phan_tram, key=thay_doi_phan_tram.get)
print(f"Thay doi tuong doi lon nhat: {thuoc_do_lon_nhat}")

"""
Chỉ một dự đoán trên tập kiểm tra được cộng thêm 5 tỷ đồng, còn các dự đoán khác
được giữ nguyên. MAE tăng theo sai số tuyệt đối thêm vào chia cho số mẫu. RMSE bình
phương sai số trước khi lấy trung bình nên sai số lớn đóng góp theo bình phương,
do đó nhạy hơn với một dự đoán lệch lớn. R2 cũng có thể thay đổi vì tổng bình phương
sai số thay đổi so với độ biến thiên của nhãn thật.
"""
print(
    f"Ket luan: {thuoc_do_lon_nhat} co thay doi tuong doi lon nhat; "
    "RMSE nhay hon voi sai so lon."
)
