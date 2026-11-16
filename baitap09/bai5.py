# -*- coding: utf-8 -*-
from time import perf_counter

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


chu_so = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(
    chu_so.data,
    chu_so.target,
    test_size=0.25,
    random_state=17,
    stratify=chu_so.target,
)

pipeline = make_pipeline(
    StandardScaler(),
    MLPClassifier(max_iter=1000, random_state=0),
)
luoi = {
    "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
    "mlpclassifier__alpha": [0.0001, 0.01, 1],
}
tim = GridSearchCV(pipeline, luoi, cv=5, n_jobs=-1)

bat_dau = perf_counter()
tim.fit(X_tr, y_tr)
thoi_gian = perf_counter() - bat_dau

bang_cv = pd.DataFrame(tim.cv_results_)
bang_cv["hidden_layer_sizes"] = bang_cv["param_mlpclassifier__hidden_layer_sizes"].map(
    str
)
bang_cv["alpha"] = bang_cv["param_mlpclassifier__alpha"].astype(float)
bang_pivot = bang_cv.pivot(
    index="hidden_layer_sizes",
    columns="alpha",
    values="mean_test_score",
).reindex(["(10,)", "(50,)", "(100,)", "(50, 50)"])

diem_mlp_grid = tim.score(X_te, y_te)
svm = SVC()
svm.fit(X_tr, y_tr)
diem_svm = accuracy_score(y_te, svm.predict(X_te))

mlp_mac_dinh = make_pipeline(
    StandardScaler(),
    MLPClassifier(max_iter=1000, random_state=0),
)
mlp_mac_dinh.fit(X_tr, y_tr)
diem_mlp_mac_dinh = accuracy_score(y_te, mlp_mac_dinh.predict(X_te))

print("Diem kiem dinh cheo trung binh (hang: hidden_layer_sizes, cot: alpha):")
print(bang_pivot.to_string(float_format=lambda gia_tri: f"{gia_tri:.4f}"))
print(f"Thoi gian GridSearchCV: {thoi_gian:.2f} giay")
print(f"Cap tham so tot nhat: {tim.best_params_}")
print(f"Diem CV tot nhat: {tim.best_score_:.4f}")
print(f"Accuracy kiem tra MLP GridSearchCV: {diem_mlp_grid:.4f}")
print(f"Accuracy kiem tra SVC mac dinh: {diem_svm:.4f}")
print(f"Accuracy kiem tra MLP mac dinh: {diem_mlp_mac_dinh:.4f}")
print(
    "Ket luan: MLP GridSearchCV "
    f"{'vuot' if diem_mlp_grid > diem_svm else 'khong vuot'} "
    f"SVC tren tap kiem tra ({diem_mlp_grid:.4f} so voi {diem_svm:.4f})."
)

"""
Cặp tham số tốt nhất là hidden_layer_sizes=(50,), alpha=1, với điểm CV 0.9785.
Accuracy kiểm tra là 0.9822 cho MLP dò lưới, 0.9911 cho SVC mặc định và 0.9844 cho MLP mặc định.
MLP dò lưới không vượt SVC trên tập kiểm tra; độ chênh giữa hai mô hình là bốn ảnh trong 450 ảnh kiểm tra.
Chênh lệch vài ảnh tương ứng với thay đổi nhỏ về accuracy nên cần diễn giải thận trọng.
"""
