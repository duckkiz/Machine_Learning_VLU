from pathlib import Path
import pandas as pd

csv_path = Path(__file__).resolve().parent / "sinh_vien.csv"
df = pd.read_csv(csv_path)

cao = df["diem_giua_ky"] >= 7
print("So ban co diem giua ky >= 7:", cao.sum())
print("Ty le qua mon nhom >= 7:", df[cao]["qua_mon"].mean())
print("Ty le qua mon nhom con lai:", df[~cao]["qua_mon"].mean())