# -*- coding: utf-8 -*-
"""Nén ảnh bằng K-Means và thống kê dung lượng tệp PNG."""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image


DUONG_DAN_ANH = "anh_cua_em.jpg"


def nen_anh(anh, rng):
    """Nén ảnh RGB về 16 màu, học trên tối đa 5000 điểm ảnh."""
    chieu_cao, chieu_rong, so_kenh = anh.shape
    anh_phang = anh.reshape(-1, so_kenh)
    so_mau = min(5000, len(anh_phang))
    chi_so_mau = rng.choice(len(anh_phang), size=so_mau, replace=False)
    mo_hinh = KMeans(n_clusters=16, n_init=4, random_state=0)
    mo_hinh.fit(anh_phang[chi_so_mau])
    nhan_mau = mo_hinh.predict(anh_phang)
    anh_nen = mo_hinh.cluster_centers_[nhan_mau].reshape(
        chieu_cao, chieu_rong, so_kenh
    )
    return np.clip(anh_nen, 0.0, 1.0)


def in_dung_luong(tep_goc, tep_nen):
    byte_goc = os.path.getsize(tep_goc)
    byte_nen = os.path.getsize(tep_nen)
    ti_le = byte_goc / byte_nen if byte_nen else float("inf")
    print(
        f"{tep_goc}: {byte_goc} byte ({byte_goc / 1024:.2f} KB); "
        f"{tep_nen}: {byte_nen} byte ({byte_nen / 1024:.2f} KB); "
        f"ti le goc/nen={ti_le:.3f}"
    )


os.makedirs("baitap08", exist_ok=True)

anh_mau = load_sample_image("china.jpg").astype(np.float32) / 255.0
anh_moi = nen_anh(anh_mau, np.random.default_rng(0))
plt.imsave("baitap08/anh_goc.png", anh_mau)
plt.imsave("baitap08/anh_16_mau.png", anh_moi)
print(f"Anh mau: {anh_mau.shape[0]} x {anh_mau.shape[1]} pixels")
in_dung_luong("baitap08/anh_goc.png", "baitap08/anh_16_mau.png")

if not os.path.exists(DUONG_DAN_ANH):
    print(
        f"Khong tim thay {DUONG_DAN_ANH}; dat anh cua ban vao thu muc goc "
        "hoac sua DUONG_DAN_ANH de bo qua phan anh ca nhan."
    )
else:
    anh_doc = np.asarray(plt.imread(DUONG_DAN_ANH))
    if anh_doc.ndim == 2:
        anh_doc = np.repeat(anh_doc[:, :, np.newaxis], 3, axis=2)
    if anh_doc.ndim != 3 or anh_doc.shape[2] not in (3, 4):
        raise ValueError("Anh ca nhan phai co 3 kenh RGB hoac 4 kenh RGBA.")
    if anh_doc.shape[2] == 4:
        anh_doc = anh_doc[:, :, :3]
    if anh_doc.dtype == np.uint8:
        anh_ca_nhan = anh_doc.astype(np.float32) / 255.0
    else:
        anh_ca_nhan = anh_doc.astype(np.float32)
    anh_ca_nhan = np.clip(anh_ca_nhan, 0.0, 1.0)
    anh_ca_nhan_nen = nen_anh(anh_ca_nhan, np.random.default_rng(0))
    plt.imsave("baitap08/anh_cua_em_goc.png", anh_ca_nhan)
    plt.imsave("baitap08/anh_cua_em_16_mau.png", anh_ca_nhan_nen)
    in_dung_luong(
        "baitap08/anh_cua_em_goc.png",
        "baitap08/anh_cua_em_16_mau.png",
    )

print(
    "Ket luan: anh RGB duoc luu bang PNG; dung luong thuc te "
    "phu thuoc vao muc do nen PNG, khong chi vao so mau."
)

"""
Ảnh mẫu 427 x 640 được lưu thành PNG gốc 550154 byte và PNG 16 màu 134431 byte, tức tệp nén nhỏ hơn khoảng 4.092 lần.
Về lý thuyết, chọn 1 trong 16 màu cần 4 bit thay cho 24 bit RGB, tương đương giảm sáu lần số bit mỗi điểm ảnh.
Hai tệp thực tế vẫn là PNG RGB và PNG đã nén sẵn, nên kích thước đo được không nhất thiết giảm đúng sáu lần.
Không tìm thấy anh_cua_em.jpg trong lần chạy này, vì vậy chương trình in hướng dẫn và bỏ qua phần ảnh cá nhân.
"""
