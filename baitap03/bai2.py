import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Đọc dữ liệu: cột noi_dung (văn bản), la_rac (1 = rác, 0 = thường)
df = pd.read_csv("tin_nhan.csv")

# 2. Biến văn bản thành ma trận đếm từ (bag of words)
vec = CountVectorizer()
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

# 3. Huấn luyện trên toàn bộ 300 tin nhắn
mo_hinh = MultinomialNB()
mo_hinh.fit(X, y)

# 4. Hiệu log-prob giữa lớp 1 (rác) và lớp 0 (thường), cho từng từ
tu = np.array(vec.get_feature_names_out())
hieu = mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]

# 5. Sắp xếp giảm dần theo hiệu
thu_tu = np.argsort(hieu)[::-1]

print("5 từ tố cáo tin RÁC mạnh nhất:")
for i in thu_tu[:5]:
    print(f"  {tu[i]:12s} hiệu = {hieu[i]:+.4f}  (tỉ lệ khả dĩ ≈ {np.exp(hieu[i]):.1f}x)")

print("\n5 từ cho thấy tin THƯỜNG mạnh nhất:")
for i in thu_tu[::-1][:5]:
    print(f"  {tu[i]:12s} hiệu = {hieu[i]:+.4f}  (tỉ lệ khả dĩ ≈ {np.exp(hieu[i]):.2f}x)")