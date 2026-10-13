import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

df = pd.read_csv("tin_nhan.csv")

# Bộ mất cân bằng: toàn bộ 180 tin thường + 20 tin rác ĐẦU TIÊN (theo thứ tự trong file)
thuong = df[df["la_rac"] == 0]
rac = df[df["la_rac"] == 1].head(20)
mat_can_bang = pd.concat([thuong, rac]).sort_index()
print("Phân bố nhãn:", mat_can_bang["la_rac"].value_counts().to_dict(),
      f"(rác chỉ chiếm {mat_can_bang['la_rac'].mean():.1%})")

def danh_gia(data, seed=42, **kw):
    a, b, c, d = train_test_split(data["noi_dung"], data["la_rac"], test_size=0.3,
                                  random_state=seed, stratify=data["la_rac"])
    vec = CountVectorizer()
    mh = MultinomialNB(**kw).fit(vec.fit_transform(a), c)
    p = mh.predict(vec.transform(b))
    return d, p, mh

d, p, mh = danh_gia(mat_can_bang)
print(f"\nTập kiểm tra: {len(d)} tin, trong đó {int(d.sum())} tin rác")
print(f"Accuracy      = {accuracy_score(d, p):.4f}")
print(f"Recall (rác)  = {recall_score(d, p, pos_label=1):.4f}")
print(f"Precision(rác)= {precision_score(d, p, pos_label=1, zero_division=0):.4f}")
print("Ma trận nhầm lẫn [hàng: thực tế 0/1, cột: dự đoán 0/1]:\n", confusion_matrix(d, p))
print(f"Mốc 'đoán toàn bộ là tin thường': accuracy = {1 - d.mean():.4f}, recall = 0")
print("Tiên nghiệm mô hình học được:", np.exp(mh.class_log_prior_).round(3))

def lap(data, **kw):
    A, R = [], []
    for s in range(50):
        d, p, _ = danh_gia(data, seed=s, **kw)
        A.append(accuracy_score(d, p)); R.append(recall_score(d, p, pos_label=1))
    return np.mean(A), np.mean(R)

print("\nTrung bình 50 lần chia:")
print("  Bộ cân bằng ban đầu (180/120)  : acc=%.3f recall=%.3f" % lap(df))
print("  Bộ mất cân bằng (180/20)       : acc=%.3f recall=%.3f" % lap(mat_can_bang))

print("  Mất cân bằng + fit_prior=False : acc=%.3f recall=%.3f" % lap(mat_can_bang, fit_prior=False))