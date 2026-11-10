# -*- coding: utf-8 -*-
"""
Bài 3 - Kernel đa thức trên dữ liệu hai vầng trăng
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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

# So sánh với RBF (tham chiếu từ tài liệu)
cv_rbf  = 0.8711
acc_rbf = 0.9200
print(f"\n  Tham chieu RBF mac dinh: CV={cv_rbf:.4f}  accuracy kiem tra={acc_rbf:.4f}")
for bac, C_b, cv, acc in ket_qua:
    so_cv  = "cao hon" if cv  > cv_rbf  else "thap hon"
    so_acc = "cao hon" if acc > acc_rbf else ("bang" if acc == acc_rbf else "thap hon")
    print(f"  Bac {bac}: CV {so_cv} RBF | accuracy kiem tra {so_acc} RBF")

print("\n  => Ket luan: kernel da thuc it khi vuot qua RBF tren du lieu vong quanh.")

"""
Nhận xét (tiếng Việt có dấu):

Kernel đa thức có thể phân chia dữ liệu hai vầng trăng nhờ đường cong bậc cao,
nhưng thường kém linh hoạt hơn kernel RBF (Gaussian) vì RBF tương đương đa thức
bậc vô hạn theo một nghĩa nào đó.

Với dữ liệu hình dạng nguyệt (moons), ranh giới phân chia có độ cong phức tạp —
kernel RBF (theo tham chiếu: CV=0.8711, accuracy=0.9200) thường vượt hoặc ngang
bằng kernel đa thức ở mọi bậc được thử.

Bậc cao hơn (bậc 4) có thể khớp dữ liệu tốt hơn trên tập học nhưng dễ bị quá khớp
(overfitting), dẫn đến accuracy kiểm tra không cải thiện hay thậm chí giảm.

Việc chuẩn hóa dữ liệu (StandardScaler) trước SVM là bắt buộc với kernel đa thức:
không chuẩn hóa thì giá trị x^d có thể phát sinh rất lớn, làm mất ổn định thuật toán.
"""

