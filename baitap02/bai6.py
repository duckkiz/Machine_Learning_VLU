import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
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

y_pred = mo_hinh.predict(X_test)

# Thứ tự bốn ô do scikit-learn quy định là TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Ma tran nham lan")
print(f" TN = {tn:2d} doan rot, that su rot")
print(f" FP = {fp:2d} doan qua, that ra rot")
print(f" FN = {fn:2d} doan rot, that ra qua")
print(f" TP = {tp:2d} doan qua, that su qua")
print()

# Lớp dương = 1 (qua môn), như ở mục 5
p1 = precision_score(y_test, y_pred)              # mặc định pos_label=1
r1 = recall_score(y_test, y_pred)
# Lớp dương = 0 (rớt môn)
p0 = precision_score(y_test, y_pred, pos_label=0)
r0 = recall_score(y_test, y_pred, pos_label=0)

print("So sanh hai cach chon lop duong:")
print(f" {'':22s}{'Precision':>10s}{'Recall':>10s}")
print(f" {'Lop 1 (qua mon)':22s}{p1:10.4f}{r1:10.4f}")
print(f" {'Lop 0 (rot mon)':22s}{p0:10.4f}{r0:10.4f}")
print()

print("Tu tinh lai cho lop 0 (doi vai tro cac o):")
print(f" Precision lop 0 = TN / (TN + FN) = {tn} / ({tn} + {fn}) = {tn / (tn + fn):.4f}")
print(f" Recall lop 0    = TN / (TN + FP) = {tn} / ({tn} + {fp}) = {tn / (tn + fp):.4f}")