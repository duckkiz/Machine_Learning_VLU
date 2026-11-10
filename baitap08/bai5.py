# -*- coding: utf-8 -*-
"""So sánh DBSCAN và K-Means trên dữ liệu hai vầng trăng."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]].to_numpy()
nhan_that = df["nhan"].to_numpy()
eps_values = [0.2, 0.3, 0.4]

# Dữ liệu được giữ nguyên thang đo để so sánh công bằng với K-Means ở mục 5.
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
nhan_kmeans = kmeans.fit_predict(X)
khop_kmeans = np.mean(nhan_kmeans == nhan_that)
if len(np.unique(nhan_that)) == 2:
    khop_kmeans = max(khop_kmeans, 1 - khop_kmeans)
print(f"KMeans khop voi nhan that: {khop_kmeans:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
ket_qua = []

for ax, eps in zip(axes, eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    nhan_cum = dbscan.fit_predict(X)
    cum_khac_nhieu = sorted(set(nhan_cum) - {-1})
    khong_nhieu = nhan_cum != -1
    so_nhieu = int(np.sum(~khong_nhieu))
    kich_thuoc = {
        int(cum): int(np.sum(nhan_cum == cum)) for cum in cum_khac_nhieu
    }

    if len(cum_khac_nhieu) == 2 and np.any(khong_nhieu):
        khop = np.mean(nhan_cum[khong_nhieu] == nhan_that[khong_nhieu])
        khop = max(khop, 1 - khop)
        khop_text = f"{khop:.4f}"
    else:
        khop = None
        khop_text = "khong ap dung"

    ket_qua.append(
        {
            "eps": eps,
            "so_cum": len(cum_khac_nhieu),
            "so_nhieu": so_nhieu,
            "kich_thuoc": kich_thuoc,
            "khop": khop,
        }
    )
    print(
        f"\neps={eps:.1f}: so cum={len(cum_khac_nhieu)}, "
        f"so diem nhieu={so_nhieu}"
    )
    print(f"Kich thuoc tung cum: {kich_thuoc}")
    print(f"Do khop tren diem khong nhieu: {khop_text}")

    for cum in cum_khac_nhieu:
        mask = nhan_cum == cum
        ax.scatter(X[mask, 0], X[mask, 1], s=18, label=f"Cum {cum}")
    if so_nhieu:
        mask_nhieu = nhan_cum == -1
        ax.scatter(
            X[mask_nhieu, 0],
            X[mask_nhieu, 1],
            s=18,
            color="gray",
            label="Nhieu",
        )
    ax.set_title(f"DBSCAN eps={eps:.1f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

plt.savefig("baitap08/bai5_hinh.png", dpi=150)
plt.close(fig)
print("\nDa luu bieu do: baitap08/bai5_hinh.png")

hop_le = [r for r in ket_qua if r["khop"] is not None]
tot_nhat = max(hop_le, key=lambda r: r["khop"]) if hop_le else None
if tot_nhat is None:
    print(
        "Ket luan: khong eps nao tao dung hai cum de tinh do khop; "
        f"KMeans khop voi nhan that {khop_kmeans:.4f}."
    )
elif tot_nhat["khop"] > khop_kmeans:
    print(
        f"Ket luan: eps={tot_nhat['eps']:.1f} co do khop "
        f"{tot_nhat['khop']:.4f}, cao hon KMeans {khop_kmeans:.4f}."
    )
else:
    print(
        "Ket luan: khong co eps nao co do khop tren diem khong nhieu "
        f"cao hon KMeans ({khop_kmeans:.4f})."
    )

"""
Với ba giá trị eps đã thử, DBSCAN đều tạo một cụm và không thể tính độ khớp hai nhãn như yêu cầu.
Ở eps=0.2 có 40 điểm nhiễu và cụm còn lại có 260 điểm; ở eps=0.3 có 10 điểm nhiễu và cụm có 290 điểm.
Ở eps=0.4 có 3 điểm nhiễu và cụm có 297 điểm, nên trong các giá trị đã thử, eps lớn hơn đi kèm ít điểm nhiễu hơn.
K-Means có độ khớp 0.7500; vì các kết quả DBSCAN không tách thành hai cụm, chưa thể kết luận DBSCAN tốt hơn K-Means theo độ khớp.
"""
