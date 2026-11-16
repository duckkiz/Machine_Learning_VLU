# -*- coding: utf-8 -*-
"""
Bài 6 - SVM trên dữ liệu lớn: đo thời gian huấn luyện
"""
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

# ── Danh sách cỡ dữ liệu cần thử ────────────────────────────────────────────
danh_sach_n = [2000, 5000, 10000, 20000]

print("=" * 65)
print("BAI 6 - SVM TREN DU LIEU LON: DO THOI GIAN HUAN LUYEN")
print("=" * 65)
print(f"  {'n':>8} | {'SVC (s)':>12} | {'LinearSVC (s)':>14} | "
      f"{'Ti le SVC':>10} | {'Ti le LSVC':>11}")
print("  " + "-" * 63)

ket_qua = []   # (n, thoi_gian_svc, thoi_gian_lsvc)

for n in danh_sach_n:
    # Tạo dữ liệu
    X, y = make_classification(n_samples=n, n_features=20, random_state=0)

    # Đo SVC (kernel RBF mặc định)
    mo_svc = SVC()
    bat_dau = time.perf_counter()
    mo_svc.fit(X, y)
    tg_svc = time.perf_counter() - bat_dau

    # Đo LinearSVC
    mo_lsvc = LinearSVC(max_iter=5000)
    bat_dau = time.perf_counter()
    mo_lsvc.fit(X, y)
    tg_lsvc = time.perf_counter() - bat_dau

    ket_qua.append((n, tg_svc, tg_lsvc))

# In bảng — tỉ lệ tăng so với cỡ liền trước
for idx, (n, tg_svc, tg_lsvc) in enumerate(ket_qua):
    if idx == 0:
        ti_le_svc  = "-"
        ti_le_lsvc = "-"
    else:
        n_prev, tg_svc_prev, tg_lsvc_prev = ket_qua[idx - 1]
        ti_le_svc  = f"{tg_svc  / tg_svc_prev :.2f}x"
        ti_le_lsvc = f"{tg_lsvc / tg_lsvc_prev:.2f}x"
    print(f"  {n:>8} | {tg_svc:>12.3f} | {tg_lsvc:>14.3f} | "
          f"{ti_le_svc:>10} | {ti_le_lsvc:>11}")

# Phân tích tỉ lệ tăng tổng thể (n_max / n_min)
n_min,  tg_svc_min,  tg_lsvc_min  = ket_qua[0]
n_max,  tg_svc_max,  tg_lsvc_max  = ket_qua[-1]
boi_so_n   = n_max / n_min        # = 10 lần

print()
print(f"  Du lieu tang {boi_so_n:.0f}x (tu {n_min} len {n_max}):")
print(f"    SVC       tang {tg_svc_max  / tg_svc_min :.1f}x thoi gian")
print(f"    LinearSVC tang {tg_lsvc_max / tg_lsvc_min:.1f}x thoi gian")

# Nhận xét khi dữ liệu tăng gấp đôi (2000→5000 không phải gấp đôi nên dùng 5000→10000)
_, tg_svc_5k,  tg_lsvc_5k  = ket_qua[1]
_, tg_svc_10k, tg_lsvc_10k = ket_qua[2]
print()
print(f"  Khi du lieu tang gap doi (5000 -> 10000):")
print(f"    SVC       tang khoang {tg_svc_10k  / tg_svc_5k :.1f}x  "
      f"(O(n^2)~O(n^3) nen tang rat nhanh)")
print(f"    LinearSVC tang khoang {tg_lsvc_10k / tg_lsvc_5k:.1f}x  "
      f"(gan tuyen tinh O(n) nen tang cham hon)")

print()
print("  => Ket luan: SVC (kernel RBF) qua cham khi n lon (do tinh ma tran")
print("     kernel n×n). LinearSVC tang tuyen tinh, phu hop du lieu lon.")

"""
Nhận xét (tiếng Việt có dấu):

SVC với kernel RBF phải tính ma trận kernel kích thước n×n (mỗi cặp điểm),
nên độ phức tạp xấp xỉ O(n²) đến O(n³) — khi dữ liệu tăng gấp đôi,
thời gian học tăng khoảng 4–8 lần hoặc hơn.

LinearSVC sử dụng tối ưu hóa tuyến tính (liblinear) không cần ma trận kernel,
độ phức tạp gần O(n) — khi dữ liệu tăng gấp đôi, thời gian học cũng
chỉ tăng khoảng 2 lần, phù hợp hơn với dữ liệu lớn.

Kết quả thời gian có thể khác nhau giữa các máy tính do tốc độ CPU, RAM và
trạng thái bộ nhớ đệm — điều quan trọng là so sánh tương đối (tỉ lệ tăng),
không phải số tuyệt đối.

Khi n = 20000, SVC có thể mất hàng chục giây đến vài phút, trong khi LinearSVC
vẫn chạy dưới vài giây, cho thấy tầm quan trọng của việc chọn thuật toán
phù hợp với quy mô dữ liệu.
"""

