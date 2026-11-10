# -*- coding: utf-8 -*-
"""Phân tích các cụm K-Means trên bộ chữ số viết tay."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score


X, nhan_that = load_digits(return_X_y=True)
mo_hinh = KMeans(n_clusters=10, n_init=10, random_state=0)
cum = mo_hinh.fit_predict(X)

bang = pd.crosstab(
    pd.Series(nhan_that, name="Chu so that"),
    pd.Series(cum, name="Cum KMeans"),
)
bang = bang.reindex(index=range(10), columns=range(10), fill_value=0)
print("Bang cheo so anh theo chu so that va cum:")
print(bang.to_string())

do_gon = {}
print("\nDo gon theo chu so (ty le trong cum lon nhat):")
for chu_so in range(10):
    dem = bang.loc[chu_so].to_numpy()
    cum_lon_nhat = int(np.argmax(dem))
    ty_le = float(dem[cum_lon_nhat] / dem.sum())
    do_gon[chu_so] = ty_le
    print(
        f"Chu so {chu_so}: cum {cum_lon_nhat}, "
        f"{dem[cum_lon_nhat]}/{dem.sum()} = {ty_le:.4f}"
    )

print("\nDo tinh khiet theo cum (chu so da so va ty le):")
do_tinh_khiet = {}
for cum_id in range(10):
    dem = bang[cum_id].to_numpy()
    chu_so_chinh = int(np.argmax(dem))
    tong = int(dem.sum())
    ty_le = float(dem[chu_so_chinh] / tong) if tong else 0.0
    do_tinh_khiet[cum_id] = (chu_so_chinh, ty_le, tong)
    print(
        f"Cum {cum_id}: chu so {chu_so_chinh}, "
        f"{dem[chu_so_chinh]}/{tong} = {ty_le:.4f}"
    )

nguong_ty_le = 0.20
print(f"\nCac cap chu so cung chiem it nhat {nguong_ty_le:.0%} trong mot cum:")
cap_da_in = 0
for cum_id in range(10):
    dem = bang[cum_id].to_numpy(dtype=float)
    tong = float(dem.sum())
    if tong == 0:
        continue
    chu_so_dang_ke = [
        chu_so for chu_so in range(10) if dem[chu_so] / tong >= nguong_ty_le
    ]
    for i, chu_so_a in enumerate(chu_so_dang_ke):
        for chu_so_b in chu_so_dang_ke[i + 1 :]:
            print(
                f"Cum {cum_id}: {chu_so_a} va {chu_so_b} "
                f"({dem[chu_so_a] / tong:.3f}, {dem[chu_so_b] / tong:.3f})"
            )
            cap_da_in += 1
if cap_da_in == 0:
    print("Khong co cap nao dat nguong.")

ari = float(adjusted_rand_score(nhan_that, cum))
chu_so_gon_nhat = max(do_gon, key=do_gon.get)
print(f"\nAdjusted Rand Index: {ari:.4f}")
print(
    "Ket luan: chu so co do gon cao nhat la "
    f"{chu_so_gon_nhat} ({do_gon[chu_so_gon_nhat]:.4f}); "
    f"ARI={ari:.4f}."
)

"""
Chữ số 0, 6 và 7 có tỷ lệ gọn lần lượt là 0.9888, 0.9779 và 0.9721; chữ số 0 cao nhất.
Theo ngưỡng 20%, chữ số 1 và 8 cùng chiếm 44.8% trong cụm 1; các cặp 1–9 và 8–9 cũng cùng chiếm đáng kể ở cụm 3 và cụm 8.
Adjusted Rand Index của kết quả này là 0.6657.
Các chữ số có nét viết gần nhau trên ảnh 8x8 có thể bị gom chung; những cặp cụ thể được nêu dựa trên bảng chéo thực tế.
"""
