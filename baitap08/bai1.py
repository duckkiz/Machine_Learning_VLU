# -*- coding: utf-8 -*-
"""Tính silhouette bằng NumPy trên hai biến đầu vào của khách vay."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/khach_vay.csv")
df = df.rename(columns={df.columns[0]: "bien_1"})
cot_phan_cum = ["bien_1", "ty_le_no"]
X_goc = df[cot_phan_cum].to_numpy(dtype=float)
scaler = StandardScaler()
Xs = scaler.fit_transform(X_goc)


def silhouette_mot_diem(X, nhan, i):
    """Tính a, b, s cho điểm i chỉ bằng NumPy."""
    khoang_cach = np.linalg.norm(X - X[i], axis=1)
    nhan_diem = nhan[i]
    cung_cum = (nhan == nhan_diem).copy()
    cung_cum[i] = False

    if not np.any(cung_cum):
        a = 0.0
    else:
        a = float(np.mean(khoang_cach[cung_cum]))

    khoang_cach_cum_khac = [
        float(np.mean(khoang_cach[nhan == cum]))
        for cum in np.unique(nhan)
        if cum != nhan_diem
    ]
    b = min(khoang_cach_cum_khac)
    if not np.any(cung_cum):
        # Silhouette của cụm chỉ có một điểm được quy ước bằng 0.
        s = 0.0
    else:
        s = (b - a) / max(a, b)
    return a, b, s


mo_hinh = KMeans(n_clusters=5, n_init=10, random_state=0)
nhan_cum = mo_hinh.fit_predict(Xs)
gia_tri = np.array(
    [silhouette_mot_diem(Xs, nhan_cum, i) for i in range(len(Xs))]
)
a_tung_diem, b_tung_diem, s_tung_diem = gia_tri.T
silhouette_tu_tinh = float(np.mean(s_tung_diem))
silhouette_thu_vien = float(silhouette_score(Xs, nhan_cum))
print(f"Silhouette trung binh tu tinh: {silhouette_tu_tinh:.6f}")
print(f"Silhouette theo sklearn: {silhouette_thu_vien:.6f}")
print(f"Hieu so hai cach tinh: {silhouette_tu_tinh - silhouette_thu_vien:.10f}")

chi_so = int(np.argmin(s_tung_diem))
nhan_diem = nhan_cum[chi_so]
khoang_cach_cac_cum = {}
for cum in np.unique(nhan_cum):
    if cum != nhan_diem:
        khoang_cach_cac_cum[cum] = float(
            np.mean(np.linalg.norm(Xs[nhan_cum == cum] - Xs[chi_so], axis=1))
        )
cum_gan_nhat = min(khoang_cach_cac_cum, key=khoang_cach_cac_cum.get)
tam_goc = scaler.inverse_transform(mo_hinh.cluster_centers_)

print(f"\nChi so dong khach co silhouette nho nhat (bat dau tu 0): {chi_so}")
print(
    "Gia tri hai bien: "
    f"bien_1={X_goc[chi_so, 0]:.4f}, ty_le_no={X_goc[chi_so, 1]:.4f}"
)
print(f"Cum cua khach: {nhan_diem}")
print(f"a={a_tung_diem[chi_so]:.6f}, b={b_tung_diem[chi_so]:.6f}, s={s_tung_diem[chi_so]:.6f}")
print(
    f"Tam cum khach (bien_1, ty_le_no): "
    f"({tam_goc[nhan_diem, 0]:.4f}, {tam_goc[nhan_diem, 1]:.4f})"
)
print(
    f"Cum gan nhat khac: {cum_gan_nhat}, tam "
    f"({tam_goc[cum_gan_nhat, 0]:.4f}, {tam_goc[cum_gan_nhat, 1]:.4f})"
)

fig, ax = plt.subplots(figsize=(8, 6))
diem = ax.scatter(
    X_goc[:, 0], X_goc[:, 1], c=nhan_cum, cmap="tab10", s=24, alpha=0.8
)
ax.scatter(
    X_goc[chi_so, 0],
    X_goc[chi_so, 1],
    marker="*",
    s=220,
    c="red",
    edgecolors="black",
    label="Silhouette nho nhat",
)
ax.set_xlabel("bien_1")
ax.set_ylabel("ty_le_no")
ax.set_title("Phan cum khach vay voi K=5")
ax.legend()
fig.colorbar(diem, ax=ax, label="Cum")
fig.tight_layout()
fig.savefig("baitap08/bai1_hinh.png", dpi=150)
plt.close(fig)

if s_tung_diem[chi_so] < 0:
    dien_giai = "diem gan cum khac hon cum cua minh"
else:
    dien_giai = "diem gan ranh gioi giua cac cum"
print(
    "Ket luan: silhouette nho nhat "
    f"{s_tung_diem[chi_so]:.4f} tai dong {chi_so}; {dien_giai}."
)

"""
Silhouette trung bình tự tính là 0.362287, bằng kết quả sklearn đến sáu chữ số thập phân.
Khách ở dòng 125 có silhouette -0.084422; a=1.354977 lớn hơn b=1.240587, nên điểm này gần cụm lân cận hơn cụm của mình.
Khách có bien_1=28.0000 và ty_le_no=0.7700; tâm cụm của khách và tâm cụm gần nhất lần lượt là (35.9311, 0.6313) và (20.1960, 0.9699).
Biểu đồ đánh dấu khách này bằng ngôi sao đỏ trên mặt phẳng hai biến đầu vào.
"""
