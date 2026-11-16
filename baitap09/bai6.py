# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


tep_du_lieu = Path("data/gia_nha.csv")
print("Cac tep trong data/:")
for tep in sorted(Path("data").iterdir()):
    if tep.is_file():
        print(tep.name)

df = pd.read_csv(tep_du_lieu)
print(f"Tep gia nha duoc chon: {tep_du_lieu}")
print("Nam dong dau:")
print(df.head().to_string(index=False))
print("Kieu du lieu:")
print(df.dtypes.to_string())
print("Thong ke mo ta:")
print(df.describe().to_string())
print("So gia tri thieu theo cot:")
print(df.isna().sum().to_string())

cot_nhan = "gia"
cot_dien_tich = "dien_tich"
cot_dau_vao = ["dien_tich", "so_phong", "tuoi_nha"]
don_vi_gia = "ty dong"
print(f"Nhan (gia): {cot_nhan}; don vi: {don_vi_gia}")
print(f"Dien tich: {cot_dien_tich} (m2)")
print("Cac cot dau vao: " + ", ".join(cot_dau_vao))

if not set(cot_dau_vao + [cot_nhan]).issubset(df.columns):
    raise ValueError("Tep gia nha khong co du cac cot can thiet.")
if df[cot_dau_vao + [cot_nhan]].isna().any().any():
    raise ValueError("Du lieu co gia tri thieu; can xu ly truoc khi hoi quy.")
if not all(pd.api.types.is_numeric_dtype(df[cot]) for cot in cot_dau_vao + [cot_nhan]):
    raise TypeError("Tep gia nha co cot khong phai so; dung lai de xac nhan schema.")

X = df[cot_dau_vao]
y = df[cot_nhan]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17
)

hoc_tuyen_tinh = make_pipeline(StandardScaler(), LinearRegression())
hoc_mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0),
)
hoc_tuyen_tinh.fit(X_tr, y_tr)
hoc_mlp.fit(X_tr, y_tr)

mae_tuyen_tinh = mean_absolute_error(y_te, hoc_tuyen_tinh.predict(X_te))
mae_mlp = mean_absolute_error(y_te, hoc_mlp.predict(X_te))
dien_tich_min = float(X_tr[cot_dien_tich].min())
dien_tich_max = float(X_tr[cot_dien_tich].max())
dien_tich_trung_vi = float(X_tr[cot_dien_tich].median())
dien_tich_250_trong_khoang = dien_tich_min <= 250 <= dien_tich_max

print(f"MAE LinearRegression: {mae_tuyen_tinh:.6f} {don_vi_gia}")
print(f"MAE MLPRegressor: {mae_mlp:.6f} {don_vi_gia}")
print(
    f"Dien tich tap hoc: min={dien_tich_min:.2f} m2, "
    f"max={dien_tich_max:.2f} m2"
)
print(
    "250 m2 nam trong khoang tap hoc: "
    f"{'Co' if dien_tich_250_trong_khoang else 'Khong'}"
)

cac_dien_tich = sorted(
    set(
        [
            dien_tich_min,
            dien_tich_trung_vi,
            dien_tich_max,
            250.0,
            350.0,
        ]
    )
)
X_du_doan = pd.DataFrame(
    {
        "dien_tich": cac_dien_tich,
        "so_phong": 4,
        "tuoi_nha": 4,
    }
)
gia_du_doan_tuyen_tinh = hoc_tuyen_tinh.predict(X_du_doan)
gia_du_doan_mlp = hoc_mlp.predict(X_du_doan)
bang_du_doan = pd.DataFrame(
    {
        "dien_tich_m2": cac_dien_tich,
        "gia_linear_ty_dong": gia_du_doan_tuyen_tinh,
        "gia_mlp_ty_dong": gia_du_doan_mlp,
    }
)
print("Du doan voi so_phong=4 va tuoi_nha=4:")
print(bang_du_doan.to_string(index=False, float_format=lambda gia_tri: f"{gia_tri:.4f}"))
print(f"Du doan gia nha 250 m2 - LinearRegression: {gia_du_doan_tuyen_tinh[cac_dien_tich.index(250.0)]:.6f} {don_vi_gia}")
print(f"Du doan gia nha 250 m2 - MLPRegressor: {gia_du_doan_mlp[cac_dien_tich.index(250.0)]:.6f} {don_vi_gia}")
print(
    "Ket luan: "
    f"{'LinearRegression' if mae_tuyen_tinh < mae_mlp else 'MLPRegressor'} "
    "co MAE thap hon tren tap kiem tra."
)

"""
LinearRegression có MAE kiểm tra 0.218546 tỷ đồng, thấp hơn MLPRegressor với 0.424246 tỷ đồng.
Diện tích trong tập học nằm từ 37.60 đến 115.90 m², nên 250 m² và 350 m² đều nằm ngoài khoảng quan sát.
Với số phòng và tuổi nhà cố định ở 4, dự đoán 250 m² lần lượt là 19.260391 và 21.664508 tỷ đồng cho LinearRegression và MLPRegressor.
Tại 350 m², hai dự đoán là 26.2810 và 30.5574 tỷ đồng; bảng cho thấy mạng dự đoán cao hơn đường thẳng ở cả hai diện tích ngoài khoảng.
Các giá trị này là ngoại suy xa khỏi diện tích học, nên không chứng minh rằng một mô hình sẽ bám sát giá thực tế bên ngoài khoảng dữ liệu.
"""
