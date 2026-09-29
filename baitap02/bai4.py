import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

csv_path = Path(__file__).resolve().parent / "sinh_vien.csv"
df = pd.read_csv(csv_path)

X = df[["gio_on"]]
y = df["qua_mon"]

# stratify=y giữ đúng tỷ lệ qua và rớt ở cả hai phần sau khi chia
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("So sinh vien de hoc :", len(X_train))
print("So sinh vien de kiem tra:", len(X_test))
print()

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

# Dự đoán nhãn trên tập kiểm tra (ngưỡng mặc định 0.5)
y_pred = mo_hinh.predict(X_test)

# Ma trận nhầm lẫn; ravel() trả về theo thứ tự TN, FP, FN, TP
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
print(f"TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}")
print()


def chia_an_toan(tu, mau):
    """Chia tu / mau, trả về 0.0 nếu mẫu số bằng 0."""
    return tu / mau if mau != 0 else 0.0


# Bốn thước đo tự tính chỉ từ TP, TN, FP, FN (lớp dương = 1 = qua môn)
accuracy = chia_an_toan(TP + TN, TP + TN + FP + FN)
precision = chia_an_toan(TP, TP + FP)
recall = chia_an_toan(TP, TP + FN)
f1 = chia_an_toan(2 * precision * recall, precision + recall)

print("Ket qua tu tinh:")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1        = {f1:.4f}")