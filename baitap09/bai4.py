# -*- coding: utf-8 -*-
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


chu_so = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(
    chu_so.data,
    chu_so.target,
    test_size=0.25,
    random_state=17,
    stratify=chu_so.target,
)

cac_toc_do = [0.0001, 0.001, 0.01, 0.1]
duong_mat_mat = []
print(
    "Learning_rate_init | n_iter | Loss cuoi | Accuracy kiem tra | "
    "Cham max_iter | Canh bao hoi tu"
)

for lr in cac_toc_do:
    mo_hinh = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100,),
            learning_rate_init=lr,
            max_iter=1000,
            random_state=0,
        ),
    )
    with warnings.catch_warnings(record=True) as canh_bao:
        warnings.simplefilter("always", ConvergenceWarning)
        mo_hinh.fit(X_tr, y_tr)

    mlp = mo_hinh.named_steps["mlpclassifier"]
    diem_kiem_tra = accuracy_score(y_te, mo_hinh.predict(X_te))
    co_canh_bao = any(
        issubclass(canh_bao_item.category, ConvergenceWarning)
        for canh_bao_item in canh_bao
    )
    cham_max_iter = mlp.n_iter_ == mlp.max_iter
    duong_mat_mat.append((lr, mlp.loss_curve_))
    print(
        f"{lr:<18g} | {mlp.n_iter_:>6} | {mlp.loss_:.6f} | "
        f"{diem_kiem_tra:.4f}            | {cham_max_iter!s:<13} | "
        f"{'Co' if co_canh_bao else 'Khong'}"
    )

plt.figure(figsize=(9, 6))
for lr, loss_curve in duong_mat_mat:
    plt.plot(
        range(1, len(loss_curve) + 1),
        loss_curve,
        label=f"learning_rate_init={lr:g}",
    )
plt.xlabel("Vong hoc")
plt.ylabel("Mat mat")
plt.title("Anh huong cua toc do hoc den mat mat")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
duong_anh = Path(__file__).resolve().parent / "toc_do_hoc.png"
plt.savefig(duong_anh, dpi=150)
plt.close()
print(f"Da luu bieu do: {duong_anh}")
print("Ket luan: So sanh bang va bieu do de danh gia tac dong cua toc do hoc.")

"""
Tốc độ 0.0001 cần 582 vòng học và có mất mát cuối 0.018928; tốc độ 0.001 cần 144 vòng và có mất mát 0.005210.
Tốc độ 0.1 dừng sau 30 vòng, đường mất mát có dao động nhìn thấy được và accuracy kiểm tra 0.9489 thấp hơn các mức còn lại.
Không tốc độ nào chạm max_iter=1000 hoặc phát sinh ConvergenceWarning trong lần chạy này.
Biểu đồ cho thấy tốc độ thấp nhất hội tụ chậm hơn, còn 0.1 giảm số vòng nhưng có mất mát cuối cao nhất trong bốn mức.
"""
