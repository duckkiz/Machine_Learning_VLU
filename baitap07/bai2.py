# -*- coding: utf-8 -*-
"""Đánh giá SVC trên bộ chữ số viết tay."""

from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


chu_so = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(
    chu_so.data,
    chu_so.target,
    test_size=0.25,
    random_state=17,
    stratify=chu_so.target,
)
mo_hinh = SVC(kernel="rbf")
mo_hinh.fit(X_tr, y_tr)
du_doan = mo_hinh.predict(X_te)

bao_cao = classification_report(
    y_te, du_doan, digits=4, output_dict=True, zero_division=0
)
print("Bao cao phan loai cho 10 chu so")
print(classification_report(y_te, du_doan, digits=4, zero_division=0))

lop = [str(so) for so in range(10)]
chu_so_recall_thap = min(lop, key=lambda so: bao_cao[so]["recall"])
chu_so_precision_thap = min(lop, key=lambda so: bao_cao[so]["precision"])
print(
    f"Chu so co recall thap nhat: {chu_so_recall_thap} "
    f"({bao_cao[chu_so_recall_thap]['recall']:.4f})"
)
print(
    f"Chu so co precision thap nhat: {chu_so_precision_thap} "
    f"({bao_cao[chu_so_precision_thap]['precision']:.4f})"
)
print("So anh moi chu so trong tap kiem tra:")
for so in range(10):
    print(f"Chu so {so}: support={int(bao_cao[str(so)]['support'])}")
print(
    f"Macro avg: precision={bao_cao['macro avg']['precision']:.4f}, "
    f"recall={bao_cao['macro avg']['recall']:.4f}, "
    f"F1={bao_cao['macro avg']['f1-score']:.4f}"
)
print(
    f"Weighted avg: precision={bao_cao['weighted avg']['precision']:.4f}, "
    f"recall={bao_cao['weighted avg']['recall']:.4f}, "
    f"F1={bao_cao['weighted avg']['f1-score']:.4f}"
)

"""
Macro avg tính trung bình đồng đều trên từng chữ số, còn weighted avg gán trọng
số theo số ảnh của mỗi chữ số. Tập kiểm tra có số ảnh giữa các lớp khá cân bằng,
nên trọng số của từng lớp gần bằng nhau. Vì vậy hai cách lấy trung bình thường
cho kết quả gần nhau; chênh lệch nhỏ vẫn có thể xuất hiện do số support không
hoàn toàn bằng nhau và hiệu năng của mô hình khác nhau giữa các lớp.
"""
print("Ket luan: macro avg va weighted avg gan nhau vi support cac lop can bang.")
