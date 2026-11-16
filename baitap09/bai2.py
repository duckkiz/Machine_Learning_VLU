# -*- coding: utf-8 -*-
import numpy as np


X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=float,
)
W1 = np.array(
    [
        [1, -1],
        [-1, 1],
    ],
    dtype=float,
)
b1 = np.zeros(2)
W2 = np.array([[1], [1]], dtype=float)
b2 = np.zeros(1)

h = np.maximum(0, X @ W1.T + b1)
y_du_doan = (h @ W2 + b2).ravel()
y_xor = np.logical_xor(X[:, 0], X[:, 1]).astype(int)

print("Bang ket qua tung noron an:")
print("x1 | x2 | h1 | h2 | y | XOR that | Dung")
for dau_vao, dau_ra_an, y, xor_that in zip(X, h, y_du_doan, y_xor):
    dung = bool(y == xor_that)
    print(
        f"{int(dau_vao[0])}  | {int(dau_vao[1])}  | "
        f"{dau_ra_an[0]:.0f}  | {dau_ra_an[1]:.0f}  | "
        f"{y:.0f} | {xor_that}        | {dung}"
    )

print("Ket qua tinh ca bon dau vao bang phep nhan ma tran:")
print("W1:")
print(W1)
print(f"b1: {b1}")
print("W2:")
print(W2)
print(f"b2: {b2}")
print(f"Y = {y_du_doan.astype(int).tolist()}")
print(f"Ket qua ca bon dong dung: {bool(np.array_equal(y_du_doan, y_xor))}")
print("Ket luan: Mang hai noron an tinh dung phep XOR tren bon dau vao.")

"""
Với đầu vào 0 hoặc 1, đầu ra y bằng |x1 - x2| nên bằng 1 khi hai đầu vào khác nhau và bằng 0 khi chúng giống nhau.
Nơ-ron ẩn thứ nhất phản hồi khi x1 lớn hơn x2, còn nơ-ron thứ hai phản hồi khi x2 lớn hơn x1.
ReLU cắt giá trị âm về 0, vì vậy chỉ một nơ-ron ẩn bật lên khi hai đầu vào khác nhau.
Khi hai đầu vào bằng nhau, cả hai nơ-ron đều cho đầu ra 0.
"""
