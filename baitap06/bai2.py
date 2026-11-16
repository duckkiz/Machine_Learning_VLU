# -*- coding: utf-8 -*-
"""
Bài 2 - So sánh SVM khi bổ sung đặc trưng chi tiêu
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


THU_MUC = Path(__file__).resolve().parent
CAC_DAC_TRUNG_CO_BAN = ["gio_on", "diem_giua_ky"]
CAC_DAC_TRUNG_MO_RONG = CAC_DAC_TRUNG_CO_BAN + ["chi_tieu"]
NHAN = "qua_mon"


def doc_du_lieu(ten_file, cac_dac_trung):
    """Đọc và kiểm tra một bộ dữ liệu sinh viên trước khi huấn luyện."""
    df = pd.read_csv(THU_MUC / ten_file)
    cac_cot_can = cac_dac_trung + [NHAN]
    cot_thieu = [cot for cot in cac_cot_can if cot not in df.columns]
    if cot_thieu:
        raise ValueError(f"{ten_file} thieu cot bat buoc: {cot_thieu}")
    if df[cac_cot_can].isna().any().any():
        raise ValueError(f"{ten_file} co gia tri thieu trong dac trung hoac nhan.")

    X = df[cac_dac_trung].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(df[NHAN], errors="raise").to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{ten_file}: nhan qua_mon phai gom ca hai lop 0 va 1.")
    return df, X, y


df_co_ban, X_co_ban, y_co_ban = doc_du_lieu(
    "sinh_vien.csv", CAC_DAC_TRUNG_CO_BAN
)
df_mo_rong, X_mo_rong, y_mo_rong = doc_du_lieu(
    "sinh_vien_mo_rong.csv", CAC_DAC_TRUNG_MO_RONG
)

if len(df_co_ban) != len(df_mo_rong):
    raise ValueError("Hai dataset phai co cung so dong de so sanh.")
if not np.array_equal(
    df_co_ban[CAC_DAC_TRUNG_CO_BAN + [NHAN]].to_numpy(),
    df_mo_rong[CAC_DAC_TRUNG_CO_BAN + [NHAN]].to_numpy(),
):
    raise ValueError(
        "Hai dataset khong dong bo theo thu tu dong o cac cot chung."
    )
if not np.array_equal(y_co_ban, y_mo_rong):
    raise ValueError("Hai dataset phai co cung nhan qua_mon.")

chi_so = np.arange(len(df_co_ban))
chi_so_tr, chi_so_te = train_test_split(
    chi_so, test_size=0.25, random_state=17, stratify=y_co_ban
)
y_tr, y_te = y_co_ban[chi_so_tr], y_co_ban[chi_so_te]

mo_hinh = {}
for ten, X, cac_dac_trung in (
    ("2 dac trung", X_co_ban, CAC_DAC_TRUNG_CO_BAN),
    ("Them chi_tieu", X_mo_rong, CAC_DAC_TRUNG_MO_RONG),
):
    phan_loai = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0),
    )
    phan_loai.fit(X[chi_so_tr], y_tr)
    du_doan = phan_loai.predict(X[chi_so_te])
    svc = phan_loai.named_steps["svc"]
    mo_hinh[ten] = {
        "model": phan_loai,
        "X_train": X[chi_so_tr],
        "X_test": X[chi_so_te],
        "predictions": du_doan,
        "accuracy": accuracy_score(y_te, du_doan),
        "support_vectors": len(svc.support_),
        "features": cac_dac_trung,
    }

print("=" * 65)
print("BAI 2 - ANH HUONG CUA DAC TRUNG CHI TIEU DEN SVM")
print("=" * 65)
print(f"  So mau: {len(df_co_ban)}  |  So mau thieu: 0")
print(f"  Phan bo nhan: {dict(zip(*np.unique(y_co_ban, return_counts=True)))}")
print(f"  Cung mot tap test cho ca hai mo hinh: {len(chi_so_te)} mau\n")
print(
    f"{'Mo hinh':<18} {'So dac trung':>13} "
    f"{'Accuracy test':>15} {'So vector ho tro':>18}"
)
print("-" * 68)
for ten, ket_qua in mo_hinh.items():
    print(
        f"{ten:<18} {len(ket_qua['features']):>13} "
        f"{ket_qua['accuracy']:>15.4f} "
        f"{ket_qua['support_vectors']:>18}"
    )
    print(f"  Ma tran nham lan (hang=that, cot=du doan):\n"
          f"{confusion_matrix(y_te, ket_qua['predictions'], labels=[0, 1])}")


def ve_mat_cat(ax, ten, ket_qua):
    """Vẽ mặt cắt SVM theo hai đặc trưng chung."""
    X_tr = ket_qua["X_train"]
    model = ket_qua["model"]
    x_min, x_max = X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1
    y_min, y_max = X_tr[:, 1].min() - 1, X_tr[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 160),
        np.linspace(y_min, y_max, 160),
    )
    luoi = np.column_stack([xx.ravel(), yy.ravel()])
    if len(ket_qua["features"]) == 3:
        chi_tieu_co_dinh = float(np.median(X_tr[:, 2]))
        luoi = np.column_stack(
            [luoi, np.full(len(luoi), chi_tieu_co_dinh)]
        )
        ghi_chu = f"chi_tieu co dinh tai trung vi {chi_tieu_co_dinh:.0f}"
    else:
        ghi_chu = "chi dung hai dac trung"

    diem = model.decision_function(luoi).reshape(xx.shape)
    ax.contour(xx, yy, diem, levels=[0], colors="black", linewidths=1.5)
    for lop, mau in ((0, "steelblue"), (1, "tomato")):
        mask = y_co_ban[chi_so_tr] == lop
        ax.scatter(
            X_tr[mask, 0],
            X_tr[mask, 1],
            c=mau,
            s=30,
            label=f"Lop {lop}",
            zorder=3,
        )
    ax.set_title(f"{ten}\n({ghi_chu})", fontsize=9)
    ax.set_xlabel(CAC_DAC_TRUNG_CO_BAN[0])
    ax.set_ylabel(CAC_DAC_TRUNG_CO_BAN[1])
    ax.legend(fontsize=8)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ve_mat_cat(axes[0], "SVM voi 2 dac trung", mo_hinh["2 dac trung"])
ve_mat_cat(axes[1], "SVM voi 3 dac trung", mo_hinh["Them chi_tieu"])
fig.suptitle("Ranh gioi phan loai; bieu do mo rong la mat cat tai chi_tieu trung vi")
fig.tight_layout()
duong_dan_hinh = THU_MUC / "bai2_hinh.png"
fig.savefig(duong_dan_hinh, dpi=120)
plt.close(fig)

chenh_lech = (
    mo_hinh["Them chi_tieu"]["accuracy"] - mo_hinh["2 dac trung"]["accuracy"]
)
if chenh_lech > 0:
    ket_luan = "tang"
elif chenh_lech < 0:
    ket_luan = "giam"
else:
    ket_luan = "khong doi"
print(
    f"\n  Them chi_tieu lam accuracy test {ket_luan} "
    f"{abs(chenh_lech):.4f} tren lan chia nay."
)
print(
    "  Luu y: ket qua chi tren mot lan chia; mat cat mo hinh 3 dac trung "
    "co dinh chi_tieu o trung vi tap huan luyen."
)
print(f"  Bieu do da luu: {duong_dan_hinh}")
