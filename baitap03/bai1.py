def bayes(priors, likelihoods):

    if len(priors) != len(likelihoods):
        raise ValueError("priors và likelihoods phải cùng độ dài")
    if abs(sum(priors) - 1.0) > 1e-9:
        raise ValueError("Tổng priors phải bằng 1")

    joint = [p * l for p, l in zip(priors, likelihoods)]

    evidence = sum(joint)
    if evidence == 0:
        raise ValueError("P(B) = 0, không thể chuẩn hóa")

    return [j / evidence for j in joint]


if __name__ == "__main__":
    nhom = ["Học đều", "Học dồn", "Ít học"]
    priors = [0.50, 0.30, 0.20]
    likelihoods = [0.95, 0.70, 0.30]

    posteriors = bayes(priors, likelihoods)

    for ten, p in zip(nhom, posteriors):
        print(f"P({ten:8s} | qua môn) = {p:.4f}")

    tong = sum(posteriors)
    print(f"\nTổng ba xác suất = {tong:.6f}")
    assert abs(tong - 1.0) < 1e-9, "Tổng không bằng 1, có lỗi!"
    print("Kiểm tra đạt: tổng bằng 1")