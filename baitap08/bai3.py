# -*- coding: utf-8 -*-
"""So sánh hai cách khởi tạo K-Means trên hai biến khách vay."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/khach_vay.csv")
df = df.rename(columns={df.columns[0]: "bien_1"})
X_goc = df[["bien_1", "ty_le_no"]].to_numpy(dtype=float)
Xs = StandardScaler().fit_transform(X_goc)
ket_qua = {}

for cach_khoi_tao in ["random", "k-means++"]:
    inertia = []
    for random_state in range(50):
        mo_hinh = KMeans(
            n_clusters=5,
            init=cach_khoi_tao,
            n_init=1,
            random_state=random_state,
        )
        mo_hinh.fit(Xs)
        inertia.append(float(mo_hinh.inertia_))
    ket_qua[cach_khoi_tao] = np.array(inertia)

muc_tot_nhat = min(float(np.min(gia_tri)) for gia_tri in ket_qua.values())
print(f"Muc inertia tot nhat trong 100 lan chay: {muc_tot_nhat:.6f}")
print("So lan dat muc tot nhat tinh voi np.isclose(atol=0.1).")

for cach_khoi_tao, inertia in ket_qua.items():
    print(f"\nKhoi tao {cach_khoi_tao}:")
    print(
        f"So lan dat: {np.isclose(inertia, muc_tot_nhat, atol=0.1).sum()}/50"
    )
    print(
        f"Nho nhat={np.min(inertia):.6f}, lon nhat={np.max(inertia):.6f}, "
        f"trung vi={np.median(inertia):.6f}"
    )
    print("Inertia da sap xep:")
    print(np.array2string(np.sort(inertia), precision=4, separator=", "))

so_dat_random = int(np.isclose(ket_qua["random"], muc_tot_nhat, atol=0.1).sum())
so_dat_plus = int(np.isclose(ket_qua["k-means++"], muc_tot_nhat, atol=0.1).sum())
if so_dat_plus > so_dat_random:
    dang_tin_cay = "k-means++"
elif so_dat_random > so_dat_plus:
    dang_tin_cay = "random"
else:
    dang_tin_cay = "hai cach bang nhau"
print(f"\nTheo so lan dat muc tot nhat, cach dang tin cay hon: {dang_tin_cay}")
print(
    "Ket luan: theo so lan dat inertia tot nhat, "
    f"{dang_tin_cay} nhieu hon (random={so_dat_random}, "
    f"k-means++={so_dat_plus}); trung vi random="
    f"{np.median(ket_qua['random']):.4f}, "
    f"k-means++={np.median(ket_qua['k-means++']):.4f}."
)

"""
Trên 50 lần chạy, khởi tạo random đạt mức inertia tốt nhất chung 7 lần, còn k-means++ đạt 6 lần; theo tiêu chí đếm này random nhỉnh hơn một lần.
Tuy vậy, trung vị inertia của k-means++ là 286.3385, thấp hơn trung vị 291.1287 của random, nên số lần chạm mức tốt nhất không kể hết khác biệt giữa các lần chạy.
Khởi tạo xấu có thể khiến K-Means dừng ở cách chia kém khi bước cập nhật không còn cải thiện inertia; thuật toán không bảo đảm tối ưu toàn cục.
K-means++ chọn tâm ban đầu cách xa nhau hơn, còn n_init lớn cho phép thử nhiều lần và giữ kết quả có inertia thấp nhất.
"""
