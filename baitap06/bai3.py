# -*- coding: utf-8 -*-
"""
Bài 3 - Kernel đa thức trên dữ liệu hai vầng trăng
"""
from pathlib import Path

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

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
print("BAI 3 - KERNEL DA THUC TREN DU LIEU HAI VANG TRANG")
print("=" * 62)
print(f"  Tap huan luyen: {len(X_tr)} mau  |  Tap kiem tra: {len(X_te)} mau\n")

# ── 3. Dò lưới cho bậc 2, 3, 4 ───────────────────────────────────────────────
danh_sach_bac = [2, 3, 4]
luoi_tham_so  = {"svc__C": [0.1, 1, 10, 100]}
ket_qua = []

for bac in danh_sach_bac:
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="poly", degree=bac)
    )
    gs = GridSearchCV(pipeline, luoi_tham_so, cv=5, n_jobs=-1)
    gs.fit(X_tr, y_tr)

    C_tot_nhat   = gs.best_params_["svc__C"]
    cv_tot_nhat  = gs.best_score_
    acc_kiem_tra = gs.score(X_te, y_te)
    ket_qua.append((bac, C_tot_nhat, cv_tot_nhat, acc_kiem_tra))

# ── 4. In bảng kết quả ───────────────────────────────────────────────────────
tieu_de = f"{'Bac':>6} | {'C tot nhat':>12} | {'CV score':>10} | {'Accuracy kiem tra':>18}"
print(tieu_de)
print("-" * len(tieu_de))
for bac, C_b, cv, acc in ket_qua:
    print(f"{bac:>6} | {C_b:>12.1f} | {cv:>10.4f} | {acc:>18.4f}")

# ── 5. Tìm bậc tốt nhất ──────────────────────────────────────────────────────
bac_tot_nhat = max(ket_qua, key=lambda r: r[2])   # theo CV score
print(f"\n  => Bac tot nhat: {bac_tot_nhat[0]} "
      f"(CV={bac_tot_nhat[2]:.4f}, accuracy kiem tra={bac_tot_nhat[3]:.4f})")

print(
    "\n  => Bac duoc chon theo diem CV, khong chon theo tap kiem tra; "
    "accuracy kiem tra chi dung de bao cao kha nang tong quat hoa."
)

"""
Nhận xét (tiếng Việt có dấu):

Kernel đa thức có thể phân chia dữ liệu hai vầng trăng nhờ đường cong bậc cao.
Kết quả CV và accuracy được tính lại trên dữ liệu hiện tại, không dùng số tham
chiếu cố định vì các số đó có thể không còn đúng sau khi cập nhật dataset.

Bậc cao hơn (bậc 4) có thể khớp dữ liệu tốt hơn trên tập học nhưng dễ bị quá khớp
(overfitting), dẫn đến accuracy kiểm tra không cải thiện hay thậm chí giảm.

Việc chuẩn hóa dữ liệu (StandardScaler) trước SVM là bắt buộc với kernel đa thức:
không chuẩn hóa thì giá trị x^d có thể phát sinh rất lớn, làm mất ổn định thuật toán.
"""

