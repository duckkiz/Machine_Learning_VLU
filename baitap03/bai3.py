import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("tin_nhan.csv")

# Chia train/test, giữ nguyên tỉ lệ rác/thường (stratify)
X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
    df["noi_dung"], df["la_rac"], test_size=0.3, random_state=42, stratify=df["la_rac"]
)

# Chỉ fit bộ đếm từ trên tập huấn luyện để tránh rò rỉ dữ liệu
vec = CountVectorizer()
X_tr = vec.fit_transform(X_tr_txt)
X_te = vec.transform(X_te_txt)

print(f"{'alpha':>8} {'accuracy':>9} {'precision':>10} {'recall':>8}")
for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
    mh = MultinomialNB(alpha=alpha).fit(X_tr, y_tr)
    pred = mh.predict(X_te)
    print(f"{alpha:>8} {accuracy_score(y_te, pred):>9.4f} "
          f"{precision_score(y_te, pred, zero_division=0):>10.4f} "
          f"{recall_score(y_te, pred, zero_division=0):>8.4f}")

# Minh họa: alpha lớn làm xác suất các từ xích lại gần nhau
print("\nĐộ chênh lệch log-prob giữa 2 lớp (max |hiệu|) theo alpha:")
for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
    mh = MultinomialNB(alpha=alpha).fit(X_tr, y_tr)
    hieu = mh.feature_log_prob_[1] - mh.feature_log_prob_[0]
    print(f"  alpha={alpha:>7}: max|hiệu| = {np.abs(hieu).max():.4f}")

print("\nĐộ ổn định qua 20 lần chia ngẫu nhiên (accuracy trung bình):")
for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
    accs, precs, recs = [], [], []
    for s in range(20):
        a, b, c, d = train_test_split(df["noi_dung"], df["la_rac"], test_size=0.3,
                                      random_state=s, stratify=df["la_rac"])
        v = CountVectorizer(); A = v.fit_transform(a); B = v.transform(b)
        p = MultinomialNB(alpha=alpha).fit(A, c).predict(B)
        accs.append(accuracy_score(d, p))
        precs.append(precision_score(d, p, zero_division=0))
        recs.append(recall_score(d, p, zero_division=0))
    print(f"  alpha={alpha:>7}: acc={np.mean(accs):.4f}  prec={np.mean(precs):.4f}  rec={np.mean(recs):.4f}")