# -*- coding: utf-8 -*-
"""
Bài 1 - Tính khoảng cách tới ranh giới SVM tuyến tính trên dữ liệu sinh viên
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DUONG_DAN_DU_LIEU = Path(__file__).resolve().parent / "sinh_vien.csv"
CAC_DAC_TRUNG = ["gio_on", "diem_giua_ky"]
NHAN = "qua_mon"

df = pd.read_csv(DUONG_DAN_DU_LIEU)
cac_cot_can = CAC_DAC_TRUNG + [NHAN]
cot_thieu = [cot for cot in cac_cot_can if cot not in df.columns]
if cot_thieu:
    raise ValueError(f"Dataset thieu cot bat buoc: {cot_thieu}")
if df[cac_cot_can].isna().any().any():
    raise ValueError("Dataset co gia tri thieu trong dac trung hoac nhan.")

X = df[CAC_DAC_TRUNG].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
y = pd.to_numeric(df[NHAN], errors="raise").to_numpy()
if set(np.unique(y)) != {0, 1}:
    raise ValueError("Nhan qua_mon phai gom ca hai lop 0 va 1.")

chi_so = np.arange(len(df))
chi_so_tr, chi_so_te = train_test_split(
    chi_so, test_size=0.25, random_state=17, stratify=y
)
X_tr, X_te = X[chi_so_tr], X[chi_so_te]
y_tr, y_te = y[chi_so_tr], y[chi_so_te]

mo_hinh = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1.0),
)
mo_hinh.fit(X_tr, y_tr)
chuan_hoa = mo_hinh.named_steps["standardscaler"]
svc = mo_hinh.named_steps["svc"]
w = svc.coef_[0]
b = svc.intercept_[0]
chuan_w = np.linalg.norm(w)

print("=" * 62)
print("BAI 1 - KHOANG CACH SVM TUYEN TINH TREN DU LIEU SINH VIEN")
print("=" * 62)
print(f"  So mau: {len(df)}  |  So mau thieu: {int(df[cac_cot_can].isna().sum().sum())}")
print(f"  Phan bo nhan: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"  Tap huan luyen: {len(X_tr)} mau  |  Tap kiem tra: {len(X_te)} mau")
print(f"  Accuracy tap kiem tra: {mo_hinh.score(X_te, y_te):.4f}")
print("\n  He so trong khong gian dac trung da chuan hoa:")
print(f"    w = ({w[0]:.4f}, {w[1]:.4f})  |  b = {b:.4f}")
print(f"    ||w|| = {chuan_w:.4f}")
print(f"    Nua do rong le ly thuyet (1/||w||) = {1.0 / chuan_w:.4f}")
print(f"    Do rong le ly thuyet (2/||w||) = {2.0 / chuan_w:.4f}")

print("\n  Khoang cach cua hai mau kiem tra dau tien:")
for vi_tri, (mau, nhan_that) in enumerate(zip(X_te[:2], y_te[:2])):
    mau_chuan_hoa = chuan_hoa.transform([mau])[0]
    f_x = float(np.dot(w, mau_chuan_hoa) + b)
    khoang_cach = abs(f_x) / chuan_w
    nhan_du_doan = int(mo_hinh.predict([mau])[0])
    print(
        f"    mau {chi_so_te[vi_tri]}: gio_on={mau[0]:.1f}, "
        f"diem_giua_ky={mau[1]:.1f}, nhan={int(nhan_that)}, "
        f"du_doan={nhan_du_doan}, f(x)={f_x:+.4f}, "
        f"khoang cach={khoang_cach:.4f}"
    )

print(f"\n  So vector ho tro: {len(svc.support_)}")
print("  Khoang cach cua vector ho tro trong khong gian da chuan hoa:")
for vi_tri, (mau, f_x) in enumerate(
    zip(svc.support_vectors_, svc.decision_function(svc.support_vectors_))
):
    khoang_cach = abs(float(f_x)) / chuan_w
    print(
        f"    mau huan luyen {chi_so_tr[svc.support_[vi_tri]]}: "
        f"f(x)={f_x:+.4f}, khoang cach={khoang_cach:.4f}"
    )

print(
    "\n  Luu y: voi SVM soft-margin, vector ho tro co the nam trong le; "
    "khong phai vector nao cung cach ranh gioi dung bang 1/||w||."
)
