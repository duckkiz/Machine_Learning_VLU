import math
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


class BoLocTuViet:
    """Naive Bayes đa thức tự viết: tự đếm từ, tự làm trơn, tính điểm bằng TỔNG LOGA."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    @staticmethod
    def tach_tu(van_ban):
        # Bỏ token 1 ký tự (như "ạ") để khớp CountVectorizer mặc định của sklearn.
        # Nếu không bỏ, từ vựng có V=40 thay vì 39 và mẫu số làm trơn sẽ lệch nhẹ.
        return [t for t in van_ban.lower().split() if len(t) >= 2]

    def fit(self, cac_tin, nhan):
        cac_tin, nhan = list(cac_tin), list(nhan)
        self.cac_lop = sorted(set(nhan))

        # 1. Từ vựng: chỉ lấy từ tập huấn luyện
        self.tu_vung = sorted({t for tin in cac_tin for t in self.tach_tu(tin)})
        V = len(self.tu_vung)

        # 2. Đếm từ theo từng lớp và đếm số tin mỗi lớp
        dem_tu = {k: Counter() for k in self.cac_lop}
        so_tin = Counter(nhan)
        for tin, k in zip(cac_tin, nhan):
            dem_tu[k].update(self.tach_tu(tin))

        # 3. Log tiên nghiệm và log khả dĩ (đã làm trơn Laplace)
        self.log_tien_nghiem = {k: math.log(so_tin[k] / len(nhan)) for k in self.cac_lop}
        self.log_kha_di = {}
        for k in self.cac_lop:
            tong = sum(dem_tu[k].values())
            mau = tong + self.alpha * V
            self.log_kha_di[k] = {
                t: math.log((dem_tu[k][t] + self.alpha) / mau) for t in self.tu_vung
            }
        return self

    def diem(self, tin):
        """Điểm mỗi lớp = log P(lớp) + Σ log P(từ | lớp)  (tổng loga, không nhân xác suất)."""
        kq = {}
        for k in self.cac_lop:
            d = self.log_tien_nghiem[k]
            for t in self.tach_tu(tin):
                if t in self.log_kha_di[k]:      # từ lạ chưa thấy khi huấn luyện thì bỏ qua
                    d += self.log_kha_di[k][t]
            kq[k] = d
        return kq

    def predict(self, cac_tin):
        return [max(self.diem(tin).items(), key=lambda kv: kv[1])[0] for tin in cac_tin]


if __name__ == "__main__":
    df = pd.read_csv("tin_nhan.csv")
    print(f"Tổng số tin nhắn: {len(df)}")

    tin_tr, tin_te, y_tr, y_te = train_test_split(
        df["noi_dung"], df["la_rac"], test_size=0.3, random_state=17, stratify=df["la_rac"]
    )


    tu_viet = BoLocTuViet(alpha=1.0).fit(tin_tr, y_tr)
    pred_tu_viet = tu_viet.predict(tin_te)


    vec = CountVectorizer()
    mh = MultinomialNB(alpha=1.0).fit(vec.fit_transform(tin_tr), y_tr)
    pred_sk = mh.predict(vec.transform(tin_te))

    acc_tu_viet = accuracy_score(y_te, pred_tu_viet)
    acc_sk = accuracy_score(y_te, pred_sk)
    print(f"Accuracy bộ lọc tự viết : {acc_tu_viet:.4f}")
    print(f"Accuracy MultinomialNB  : {acc_sk:.4f}")
    print(f"Độ lệch                 : {abs(acc_tu_viet - acc_sk):.4f}")
    print(f"Số tin hai bên dự đoán khác nhau: {int(np.sum(np.array(pred_tu_viet) != pred_sk))}/{len(y_te)}")

    # Kiểm tra sâu hơn: so sánh trực tiếp log khả dĩ với feature_log_prob_
    chi_so = {t: i for i, t in enumerate(vec.get_feature_names_out())}
    lech = max(
        abs(tu_viet.log_kha_di[k][t] - mh.feature_log_prob_[i][chi_so[t]])
        for i, k in enumerate(tu_viet.cac_lop) for t in chi_so
    )
    print(f"Sai lệch lớn nhất của log khả dĩ so với sklearn: {lech:.2e}")