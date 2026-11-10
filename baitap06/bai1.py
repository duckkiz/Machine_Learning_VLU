# -*- coding: utf-8 -*-
"""
Bài 1 - Tính tay khoảng cách tới ranh giới SVM tuyến tính
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# ── 1. Tạo lại 24 điểm như trong tài liệu ──────────────────────────────────
X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                  cluster_std=0.9, random_state=0)
y = 1 - y   # đảo nhãn: lớp 0 ↔ lớp 1

# ── 2. Huấn luyện SVM tuyến tính ────────────────────────────────────────────
mo_hinh = SVC(kernel="linear", C=1000)
mo_hinh.fit(X, y)

w1 = mo_hinh.coef_[0][0]
w2 = mo_hinh.coef_[0][1]
b  = mo_hinh.intercept_[0]

print("=" * 55)
print("BAI 1 - TINH TAY KHOANG CACH TOI RANH GIOI SVM")
print("=" * 55)
print(f"  w1 = {w1:.4f}  |  w2 = {w2:.4f}  |  b = {b:.4f}")

# ── 3. Tính f(x) = w1*x1 + w2*x2 + b bằng tay ──────────────────────────────
diem_A = np.array([3.0, 3.0])
diem_B = np.array([5.0, 4.5])

f_A = w1 * diem_A[0] + w2 * diem_A[1] + b
f_B = w1 * diem_B[0] + w2 * diem_B[1] + b

print(f"\n  f(3, 3)     (tinh tay) = {f_A:+.4f}")
print(f"  f(5, 4.5)   (tinh tay) = {f_B:+.4f}")

# Đối chiếu với decision_function (chỉ in để kiểm tra, không dùng để tính)
df_A = mo_hinh.decision_function([diem_A])[0]
df_B = mo_hinh.decision_function([diem_B])[0]
print(f"\n  [doi chieu] decision_function(3,3)   = {df_A:+.4f}")
print(f"  [doi chieu] decision_function(5,4.5) = {df_B:+.4f}")

# ── 4. Khoảng cách tới ranh giới = |f(x)| / ||w|| ──────────────────────────
chuan_w = np.sqrt(w1**2 + w2**2)   # ||w||
print(f"\n  ||w|| = sqrt(w1^2 + w2^2) = {chuan_w:.4f}")

kc_A = abs(f_A) / chuan_w
kc_B = abs(f_B) / chuan_w
print(f"\n  Khoang cach (3, 3)   toi ranh gioi = {kc_A:.4f}")
print(f"  Khoang cach (5, 4.5) toi ranh gioi = {kc_B:.4f}")

# ── 5. Nửa độ rộng lề = 1 / ||w|| ──────────────────────────────────────────
nua_le = 1.0 / chuan_w
do_rong_le = 2.0 / chuan_w
print(f"\n  Nua do rong le (1/||w||) = {nua_le:.4f}")
print(f"  Do rong le     (2/||w||) = {do_rong_le:.4f}")

# ── 6. Khoảng cách của từng vectơ hỗ trợ ────────────────────────────────────
chi_so_vt = mo_hinh.support_          # chỉ số trong tập dữ liệu
so_vt = len(chi_so_vt)
print(f"\n  So vecto ho tro: {so_vt}  |  chi so: {chi_so_vt.tolist()}")
print("  Khoang cach tung vecto ho tro:")
for idx in chi_so_vt:
    xi = X[idx]
    f_xi = w1 * xi[0] + w2 * xi[1] + b
    kc_xi = abs(f_xi) / chuan_w
    print(f"    diem {idx:3d}  x=({xi[0]:.3f},{xi[1]:.3f})  "
          f"f(x)={f_xi:+.4f}  kc={kc_xi:.4f}  (nua le={nua_le:.4f})")

print("\n  => Khoang cach cua vecto ho tro bang dung nua do rong le.")
print("  => Ket luan: vecto ho tro nam tren mep le nen |f(x)| = 1 chinh xac.")

"""
Nhận xét (tiếng Việt có dấu):

Các vectơ hỗ trợ là những điểm nằm chính xác trên mép của lề, nghĩa là
tại đó |f(x)| = w·x + b = ±1 (theo quy ước chuẩn của SVM).

Khoảng cách từ một điểm bất kỳ tới siêu phẳng là |f(x)| / ||w||.
Vì |f(x)| = 1 với các vectơ hỗ trợ, khoảng cách của chúng bằng 1/||w||,
đúng bằng nửa độ rộng lề (margin = 2/||w||, nên nửa lề = 1/||w||).

Điểm (3,3) có f > 0 → thuộc lớp dương, khoảng cách lớn hơn nửa lề vì
nó không phải vectơ hỗ trợ. Điểm (5,4.5) có f < 0 → thuộc lớp âm.

Đây là lý do tại sao SVM tối ưu hoá sao cho 1/||w|| lớn nhất (tức ||w||
nhỏ nhất): muốn lề rộng nhất thì phải ép ||w|| nhỏ nhất, trong khi
vẫn đảm bảo các vectơ hỗ trợ nằm đúng trên mép |f| = 1.
"""

