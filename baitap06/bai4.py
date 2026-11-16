# -*- coding: utf-8 -*-
"""
Bài 4 - Dò lưới mịn trên hai vầng trăng (kernel RBF)
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── 1. Nạp dữ liệu ───────────────────────────────────────────────────────────
DUONG_DAN_DU_LIEU = Path(__file__).resolve().parent / "hai_vang_trang.xlsx"
df = pd.read_excel(DUONG_DAN_DU_LIEU)
cac_cot_can = ["x1", "x2", "nhan"]
cot_thieu = [cot for cot in cac_cot_can if cot not in df.columns]
if cot_thieu:
    raise ValueError(f"Dataset thieu cot bat buoc: {cot_thieu}")
if df[cac_cot_can].isna().any().any():
    raise ValueError("Dataset co gia tri thieu trong dac trung hoac nhan.")
X = df[["x1", "x2"]].apply(pd.to_numeric, errors="raise").to_numpy()
y = pd.to_numeric(df["nhan"], errors="raise").to_numpy()

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
luoi_tham_so = {"svc__C": danh_sach_C, "svc__gamma": danh_sach_gamma}

pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
gs = GridSearchCV(pipeline, luoi_tham_so, cv=5, n_jobs=-1)
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
C_tot     = gs.best_params_["svc__C"]
gamma_tot = gs.best_params_["svc__gamma"]
cv_tot    = gs.best_score_
acc_te    = gs.score(X_te, y_te)

print(f"\n  Cap tot nhat: C={C_tot}  gamma={gamma_tot}")
print(f"  Diem CV tot nhat:    {cv_tot:.4f}")
print(f"  Accuracy tap kiem tra: {acc_te:.4f}")

so_to_hop = len(danh_sach_C) * len(danh_sach_gamma)
print(f"\n  So to hop tham so: {so_to_hop}  |  So lan huan luyen: {so_to_hop * 5} (cv=5)")
print(
    "  => Ket luan: ket qua duoc tinh lai tren dataset hien tai; "
    "chuan hoa duoc fit rieng trong tung fold CV."
)

"""
Nhận xét (tiếng Việt có dấu):

Dò lưới mịn thử các giá trị C và gamma trên dataset hiện tại. Pipeline đặt
StandardScaler bên trong GridSearchCV để việc chuẩn hóa chỉ được fit trên phần
huấn luyện của từng fold, tránh rò rỉ thông tin từ fold kiểm định.

Số lần huấn luyện tăng lên theo số tổ hợp × số fold (ở đây 9 × 5 = 45 lần).

Điều này nhắc nhở rằng: sau một mức độ tối ưu nhất định, việc dò lưới quá tỉ mỉ chỉ
tốn thêm thời gian mà không mang lại mô hình tốt hơn trên dữ liệu mới (ngoài mẫu).
Thay vào đó nên tập trung vào chất lượng dữ liệu và lựa chọn đặc trưng.
"""

