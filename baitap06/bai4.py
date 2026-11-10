# -*- coding: utf-8 -*-
"""
Bài 4 - Dò lưới mịn trên hai vầng trăng (kernel RBF)
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

# ── 1. Nạp dữ liệu ───────────────────────────────────────────────────────────
df = pd.read_excel("baitap06/hai_vang_trang.xlsx")
X = df[["x1", "x2"]].values
y = df["nhan"].values

# ── 2. Chia train / test ──────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("=" * 62)
print("BAI 4 - DO LUOI MIN TREN HAI VANG TRANG (KERNEL RBF)")
print("=" * 62)
print(f"  Tap huan luyen: {len(X_tr)} mau  |  Tap kiem tra: {len(X_te)} mau\n")

# ── 3. Dò lưới mịn ───────────────────────────────────────────────────────────
danh_sach_C     = [3, 10, 30]
danh_sach_gamma = [0.3, 1, 3]
luoi_tham_so = {"C": danh_sach_C, "gamma": danh_sach_gamma}

gs = GridSearchCV(SVC(kernel="rbf"), luoi_tham_so, cv=5, n_jobs=-1)
gs.fit(X_tr, y_tr)

# ── 4. In bảng pivot điểm CV ─────────────────────────────────────────────────
print("  Bang diem CV (hang=C, cot=gamma):")

# Tiêu đề cột
tieu_de_cot = "       C |" + "".join(f"  gamma={g:>4}" for g in danh_sach_gamma)
print(f"  {tieu_de_cot}")
print("  " + "-" * (len(tieu_de_cot) + 2))

ket_qua_cv = gs.cv_results_["mean_test_score"].reshape(
    len(danh_sach_C), len(danh_sach_gamma)
)
for i, C in enumerate(danh_sach_C):
    dong = f"  C={C:>5} |" + "".join(f"  {ket_qua_cv[i, j]:>10.4f}" for j in range(len(danh_sach_gamma)))
    print(dong)

# ── 5. Cặp tốt nhất và accuracy ──────────────────────────────────────────────
C_tot     = gs.best_params_["C"]
gamma_tot = gs.best_params_["gamma"]
cv_tot    = gs.best_score_
acc_te    = gs.score(X_te, y_te)

print(f"\n  Cap tot nhat: C={C_tot}  gamma={gamma_tot}")
print(f"  Diem CV tot nhat:    {cv_tot:.4f}")
print(f"  Accuracy tap kiem tra: {acc_te:.4f}")

# ── 6. So sánh với tham chiếu (mục 5 tài liệu) ───────────────────────────────
cv_ref  = 0.8711
acc_ref = 0.9200
print(f"\n  Tham chieu (luoi thu): CV={cv_ref:.4f}  accuracy={acc_ref:.4f}")
print(f"  Luoi min:              CV={cv_tot:.4f}  accuracy={acc_te:.4f}")
if cv_tot > cv_ref:
    print("  => CV tot nhat luoi min CAO HON luoi thu (0.8711).")
elif cv_tot == cv_ref:
    print("  => CV tot nhat luoi min BANG luoi thu.")
else:
    print("  => CV tot nhat luoi min THAP HON luoi thu — do luoi khac khong bao gom cac gia tri toi uu.")

if abs(acc_te - acc_ref) < 1e-4:
    print("  => Accuracy kiem tra KHONG DOI so voi luoi thu.")
else:
    print(f"  => Accuracy kiem tra {'TANG' if acc_te > acc_ref else 'GIAM'} so voi luoi thu.")

so_to_hop = len(danh_sach_C) * len(danh_sach_gamma)
print(f"\n  So to hop tham so: {so_to_hop}  |  So lan huan luyen: {so_to_hop * 5} (cv=5)")
print("  => Ket luan: do luoi min tang so lan hoc nhieu nhung loi ich thuong rat nho.")

"""
Nhận xét (tiếng Việt có dấu):

Dò lưới mịn tức là chọn các giá trị C và gamma nằm giữa các bước của lưới thưa ban đầu,
hy vọng tìm được tổ hợp cho điểm CV cao hơn. Tuy nhiên, trên thực tế, lợi ích thường
rất nhỏ và đôi khi chỉ là do nhiễu từ quá trình kiểm định chéo.

Số lần huấn luyện tăng lên theo số tổ hợp × số fold (ở đây 9 × 5 = 45 lần), trong khi
accuracy trên tập kiểm tra thường không cải thiện đáng kể so với lưới thưa.

Điều này nhắc nhở rằng: sau một mức độ tối ưu nhất định, việc dò lưới quá tỉ mỉ chỉ
tốn thêm thời gian mà không mang lại mô hình tốt hơn trên dữ liệu mới (ngoài mẫu).
Thay vào đó nên tập trung vào chất lượng dữ liệu và lựa chọn đặc trưng.
"""

