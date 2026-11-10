# -*- coding: utf-8 -*-
"""
Bài 2 - Thêm một điểm mới và quan sát ảnh hưởng lên SVM
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# ── Hàm tiện ích ─────────────────────────────────────────────────────────────
def hoc_va_in(X_train, y_train, ten_truong_hop):
    """Học SVM tuyến tính và trả về (w1, w2, b, do_rong_le, chi_so_vt)."""
    mo = SVC(kernel="linear", C=1000)
    mo.fit(X_train, y_train)
    w1 = mo.coef_[0][0]
    w2 = mo.coef_[0][1]
    b  = mo.intercept_[0]
    chuan_w = np.sqrt(w1**2 + w2**2)
    do_rong_le = 2.0 / chuan_w
    chi_so_vt  = mo.support_.tolist()
    print(f"\n--- {ten_truong_hop} ---")
    print(f"  w1={w1:.4f}  w2={w2:.4f}  b={b:.4f}")
    print(f"  Do rong le = {do_rong_le:.4f}")
    print(f"  So vecto ho tro: {len(chi_so_vt)}  chi so: {chi_so_vt}")
    return mo, w1, w2, b, do_rong_le, chi_so_vt

def ve_bieu_do(ax, X_train, y_train, mo, ten, diem_them=None, nhan_them=None):
    """Vẽ điểm dữ liệu và đường phân chia lên axes."""
    mau = ["steelblue", "tomato"]
    for lop in [0, 1]:
        mask = y_train == lop
        ax.scatter(X_train[mask, 0], X_train[mask, 1],
                   c=mau[lop], s=40, label=f"Lop {lop}", zorder=3)
    # Đánh dấu vectơ hỗ trợ
    vt = mo.support_vectors_
    ax.scatter(vt[:, 0], vt[:, 1], s=120, facecolors="none",
               edgecolors="k", linewidths=1.5, label="Vecto ho tro", zorder=4)
    # Điểm thêm mới
    if diem_them is not None:
        mau_them = "blue" if nhan_them == 1 else "red"
        ax.scatter(*diem_them, c=mau_them, marker="*", s=200,
                   edgecolors="k", zorder=5, label="Diem them")
    # Đường ranh giới và lề
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    xx = np.linspace(x_min, x_max, 200)
    w1, w2, b = mo.coef_[0][0], mo.coef_[0][1], mo.intercept_[0]
    if abs(w2) > 1e-9:
        ax.plot(xx, -(w1 * xx + b) / w2, "k-", lw=1.5)
        ax.plot(xx, -(w1 * xx + b + 1) / w2, "k--", lw=0.8, alpha=0.6)
        ax.plot(xx, -(w1 * xx + b - 1) / w2, "k--", lw=0.8, alpha=0.6)
    ax.set_title(ten, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(x_min, x_max)

# ── 1. Dữ liệu gốc 24 điểm ──────────────────────────────────────────────────
X0, y0 = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                    cluster_std=0.9, random_state=0)
y0 = 1 - y0

# Giá trị gốc tham chiếu
w1_goc, w2_goc, b_goc = -0.8314, -0.3549, 4.8204

print("=" * 60)
print("BAI 2 - THEM MOT DIEM VAO TAP HUAN LUYEN")
print("=" * 60)

mo_goc, w1g, w2g, bg, le_goc, vt_goc = hoc_va_in(X0, y0, "GOC (24 diem)")

# ── 2. Lần 1: thêm điểm lớp 1 tại (-1, -1) ─────────────────────────────────
diem_1 = np.array([[-1.0, -1.0]])
X_lan1 = np.vstack([X0, diem_1])
y_lan1 = np.append(y0, [1])

mo_1, w1_1, w2_1, b_1, le_1, vt_1 = hoc_va_in(X_lan1, y_lan1,
                                                "LAN 1: them diem lop 1 tai (-1,-1)")
print(f"  Hieu so voi goc: Dw1={w1_1-w1g:+.4f} Dw2={w2_1-w2g:+.4f} Db={b_1-bg:+.4f}")
diem_1_la_vt = (len(X0) in vt_1)   # chỉ số 24 (điểm mới)
print(f"  Diem (-1,-1) co la vecto ho tro? {'CO' if diem_1_la_vt else 'KHONG'}")

# ── 3. Lần 2: thêm điểm lớp 1 tại (4.5, 4) ─────────────────────────────────
diem_2 = np.array([[4.5, 4.0]])
X_lan2 = np.vstack([X0, diem_2])
y_lan2 = np.append(y0, [1])

mo_2, w1_2, w2_2, b_2, le_2, vt_2 = hoc_va_in(X_lan2, y_lan2,
                                                "LAN 2: them diem lop 1 tai (4.5,4)")
print(f"  Hieu so voi goc: Dw1={w1_2-w1g:+.4f} Dw2={w2_2-w2g:+.4f} Db={b_2-bg:+.4f}")
diem_2_la_vt = (len(X0) in vt_2)
print(f"  Diem (4.5,4) co la vecto ho tro? {'CO' if diem_2_la_vt else 'KHONG'}")

# Kiểm tra tách tuyến tính
# Nếu số vectơ hỗ trợ từ cả hai lớp, dữ liệu còn có thể tách bằng đường thẳng
# Với C=1000 (rất lớn) → phạt lỗi nặng → ép phân tách hoàn toàn nếu có thể
n_sai_lan2 = sum(mo_2.predict(X_lan2) != y_lan2)
print(f"  So diem phan loai sai tren tap huan luyen (C=1000): {n_sai_lan2}")
print(f"  => Du lieu {'CON' if n_sai_lan2==0 else 'KHONG'} tach duoc hoan toan bang duong thang")

print("\n  => Lan 1: duong ranh gioi gan nhu khong doi (diem (-1,-1) nam sau trong vung lop 1).")
print("  => Lan 2: duong ranh gioi thay doi ro (diem (4.5,4) lop 1 lan sang phia lop 0).")
print("  => Ket luan: chi vecto ho tro moi anh huong duong ranh gioi; C=1000 nen SVM co")
print("     gang tach hoan toan, duong ranh gioi se bi keo khi co diem kho phan tach.")

# ── 4. Vẽ 3 biểu đồ ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ve_bieu_do(axes[0], X0,     y0,     mo_goc, "Goc (24 diem)")
ve_bieu_do(axes[1], X_lan1, y_lan1, mo_1,   "Lan 1: them (-1,-1) lop 1",
           diem_them=(-1, -1), nhan_them=1)
ve_bieu_do(axes[2], X_lan2, y_lan2, mo_2,   "Lan 2: them (4.5,4) lop 1",
           diem_them=(4.5, 4), nhan_them=1)
plt.tight_layout()
plt.savefig("baitap06/bai2_hinh.png", dpi=120)
plt.close()
print("\n  Bieu do da luu: baitap06/bai2_hinh.png")

"""
Nhận xét (tiếng Việt có dấu):

Lần 1 (thêm điểm (-1,-1) lớp 1): Điểm này nằm rất xa phía lớp 1, sâu trong vùng an toàn của
lớp đó. Nó không thể trở thành vectơ hỗ trợ vì khoảng cách của nó đến ranh giới lớn hơn nhiều
so với lề. Do vậy, đường phân chia gần như không thay đổi so với ban đầu (hiệu số w và b ≈ 0).

Lần 2 (thêm điểm (4.5, 4) lớp 1): Điểm này nằm giữa hai đám mây, lấn sang phía lớp 0.
Nó trở thành vectơ hỗ trợ và "kéo" đường ranh giới về phía mình, làm thay đổi đáng kể
w1, w2 và b so với đường gốc. SVM với C=1000 rất lớn nên cố ép tách hoàn toàn — nếu dữ liệu
vẫn tách tuyến tính được thì vẫn không có lỗi, nhưng lề bị thu hẹp.

Điều này minh họa nguyên lý cốt lõi: **chỉ những vectơ hỗ trợ mới quyết định đường ranh giới**;
những điểm nằm sâu bên trong vùng lớp của mình hoàn toàn không ảnh hưởng.
"""

