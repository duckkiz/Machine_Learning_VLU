# -*- coding: utf-8 -*-
"""So sánh phân cụm hai biến và bốn biến đầu vào của khách vay."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/khach_vay.csv")
df = df.rename(columns={df.columns[0]: "bien_1"})
cot_co_ban = ["bien_1", "ty_le_no"]
cot_dau_vao = [
    "bien_1",
    "ty_le_no",
    "co_viec_on_dinh",
    "so_nguoi_phu_thuoc",
]


def danh_gia(cot):
    X_goc = df[cot].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_goc)
    ket_qua = []
    for k in range(2, 11):
        mo_hinh = KMeans(n_clusters=k, n_init=10, random_state=0)
        nhan_cum = mo_hinh.fit_predict(Xs)
        silhouette = silhouette_score(Xs, nhan_cum)
        ket_qua.append((k, mo_hinh.inertia_, silhouette))
        print(
            f"K={k}: inertia={mo_hinh.inertia_:.4f}, "
            f"silhouette={silhouette:.4f}"
        )
    k_tot_nhat = max(ket_qua, key=lambda muc: muc[2])[0]
    print(f"K tot nhat theo silhouette: {k_tot_nhat}")
    return X_goc, scaler, k_tot_nhat, ket_qua


print("=== Hai bien co ban: bien_1, ty_le_no ===")
_, _, _, ket_qua_co_ban = danh_gia(cot_co_ban)
silhouette_co_ban_k5 = next(muc[2] for muc in ket_qua_co_ban if muc[0] == 5)

print("\n=== Bon bien dau vao (khong dung tra_dung_han) ===")
X_goc, scaler, k_tot_nhat, ket_qua_day_du = danh_gia(cot_dau_vao)
silhouette_day_du_k5 = next(muc[2] for muc in ket_qua_day_du if muc[0] == 5)

for k, tieu_de in [(k_tot_nhat, "K tot nhat"), (5, "K=5")]:
    mo_hinh = KMeans(n_clusters=k, n_init=10, random_state=0)
    nhan_cum = mo_hinh.fit_predict(scaler.transform(X_goc))
    tam_goc = scaler.inverse_transform(mo_hinh.cluster_centers_)
    kich_thuoc = np.bincount(nhan_cum, minlength=k)
    thu_tu = np.argsort(tam_goc[:, 1])
    print(f"\nTam cum {tieu_de}, sap theo ty_le_no:")
    print(
        "Cum | bien_1 | ty_le_no | co_viec_on_dinh | "
        "so_nguoi_phu_thuoc | so_khach"
    )
    for cum in thu_tu:
        print(
            f"{cum:3d} | {tam_goc[cum, 0]:7.3f} | {tam_goc[cum, 1]:8.3f} | "
            f"{tam_goc[cum, 2]:16.3f} | {tam_goc[cum, 3]:19.3f} | "
            f"{kich_thuoc[cum]:8d}"
        )

hieu_so = silhouette_day_du_k5 - silhouette_co_ban_k5
print(
    f"\nSilhouette K=5, hai bien: {silhouette_co_ban_k5:.4f}; "
    f"bon bien: {silhouette_day_du_k5:.4f}; chenh lech: {hieu_so:+.4f}"
)
print(
    "Ket luan: "
    f"K tot nhat voi bon bien la {k_tot_nhat}; "
    f"silhouette K=5 thay doi {hieu_so:+.4f} khi them cac bien dau vao."
)

"""
Tệp dữ liệu không có tuổi, thu nhập hoặc điểm chi tiêu; với bốn biến đầu vào, silhouette cao nhất trong K=2..10 đạt 0.2968 tại K=7.
Với hai biến đầu vào, K=3 đạt silhouette cao nhất 0.4145; ở K=5, silhouette giảm từ 0.3623 xuống 0.2658 khi dùng cả bốn biến.
Ở K=7, các tâm có cùng tỷ lệ nợ xấp xỉ 0.321 nhưng khác trạng thái việc làm ổn định, lần lượt là 0.000 và 1.000.
Nhãn tra_dung_han không được dùng để tạo cụm; các tâm cho thấy thêm thông tin về việc làm và số người phụ thuộc.
"""
