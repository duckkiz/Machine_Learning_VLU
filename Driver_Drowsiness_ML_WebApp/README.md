# HỆ THỐNG CẢNH BÁO TÀI XẾ BUỒN NGỦ & MẤT TẬP TRUNG (WEB APP)
### *Driver Drowsiness & Inattention Detection Web Application*
> **Đồ án môn học:** Học Máy (Machine Learning)  
> **Phiên bản:** 2.0 (Web Deployment & Real-time Inference)

---

## 📌 Giới thiệu đề tài
Hệ thống kết hợp kỹ thuật thị giác máy tính và học sâu (Deep Learning / Machine Learning) nhằm giám sát trạng thái lái xe theo thời gian thực, kịp thời phát hiện và cảnh báo các dấu hiệu nguy hiểm:
- 🥱 **Buồn ngủ (Drowsy):** Mắt nhắm, gục đầu.
- 😮 **Ngáp (Yawning):** Miệng mở to trong khoảng thời gian nhất định.
- 👀 **Mất tập trung (Looking Away):** Quay mặt khỏi hướng nhìn phía trước.
- 🟢 **Tỉnh táo (Awake):** Trạng thái an toàn bình thường.

Ứng dụng được đóng gói hoàn chỉnh dưới dạng **Web Application** với giao diện Dark Glassmorphism trực quan, hỗ trợ:
1. **Dashboard:** Thống kê độ chính xác (98.8%), thông tin mô hình, luồng pipeline.
2. **Evaluation Results:** Trực quan hóa ma trận nhầm lẫn (Confusion Matrix), biểu đồ Accuracy/Loss curves, báo cáo phân lớp chi tiết.
3. **Live Camera Demo:** Nhận diện trực tiếp qua Webcam với cơ chế chống nhiễu (Temporal Voting) và cảnh báo âm thanh/hình ảnh.

---

## 🧠 Kiến trúc Pipeline & Mô hình Học Máy

```
[Webcam Stream / Camera]
          │
          ▼
[MediaPipe Face Landmarker] ──► (Trích xuất 478 điểm mốc & Cắt ROI khuôn mặt)
          │
          ▼
[YOLOv11-cls Classification] ──► (Dự đoán xác suất 4 lớp trạng thái)
          │
          ▼
[Sliding Temporal Voting Window] ──► (Bộ lọc đa số phiếu qua 9 khung hình để khử giật)
          │
          ▼
[Flask Web Interface / MJPEG Stream & Audio Alert]
```

### 1. Trích xuất đặc trưng khuôn mặt (Face ROI)
- Sử dụng **MediaPipe Face Landmarker** phát hiện 478 tọa độ mốc 3D của khuôn mặt.
- Tự động căn lề (margin) và crop chính xác vùng mặt nhằm loại bỏ nhiễu từ bối cảnh xung quanh xe.

### 2. Mô hình phân loại trạng thái (YOLOv11 Classification)
- Kiến trúc: **YOLO11-cls (Nano)** được huấn luyện trên bộ dữ liệu khuôn mặt tài xế.
- Độ chính xác kiểm thử (**Test Accuracy**): **98.8%**
- **Macro F1-Score**: **0.988**
- Tốc độ suy luận: **~30 - 45 FPS** trên CPU / GPU thông thường.

### 3. Cơ chế ổn định nhãn (Temporal Voting Window)
- Sử dụng cửa sổ trượt kích thước $N = 9$ khung hình gần nhất.
- Yêu cầu tối thiểu $K = 6$ phiếu đồng thuận trước khi kích hoạt cảnh báo, loại bỏ hoàn toàn hiện tượng nhấp nháy hoặc phát hiện nhầm khi tài xế chớp mắt tự nhiên.

---

## 📁 Cấu trúc thư mục dự án

```
Driver_Drowsiness_ML_WebApp/
├── app.py                      # Server Flask chính (REST API + MJPEG Streaming)
├── config.py                   # Cấu hình đường dẫn và thông số mô hình
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── README.md                   # Hướng dẫn chi tiết dự án
│
├── models/                     # Trọng số mô hình đã huấn luyện
│   ├── best.pt                 # Trọng số YOLOv11 phân loại trạng thái
│   ├── face_landmarker.task    # Model trích xuất mốc khuôn mặt MediaPipe
│   └── yolo11n-cls.pt          # Backbone weights
│
├── templates/                  # Giao diện HTML (Jinja2)
│   ├── base.html               # Khung layout dùng chung & thanh điều hướng
│   ├── index.html              # Trang Dashboard tổng quan
│   ├── results.html            # Trang Kết quả thực nghiệm & Ma trận nhầm lẫn
│   └── demo.html               # Trang Live Demo Webcam thời gian thực
│
├── static/                     # Tài nguyên tĩnh
│   ├── css/
│   │   └── style.css           # Giao diện Dark Glassmorphism UI
│   └── js/
│       └── main.js             # Client scripts & hiệu ứng tương tác
│
├── outputs/                    # Báo cáo và biểu đồ đánh giá
│   ├── figures/                # Biểu đồ Accuracy, Loss, Confusion Matrix
│   └── reports/                # Báo cáo số liệu JSON, TXT, CSV
│
└── scripts/                    # Các script mở rộng & phụ trợ
    ├── detect_webcam.py        # Chạy kiểm thử webcam trên OpenCV thuần
    ├── evaluate.py             # Đánh giá mô hình trên tập kiểm thử
    ├── train.py                # Script huấn luyện lại mô hình YOLO
    └── prepare_dataset.py      # Tiền xử lý dữ liệu và trích xuất ROI
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Mở Terminal tại thư mục dự án
```bash
cd Driver_Drowsiness_ML_WebApp
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```
*(Các thư viện chính bao gồm: `flask`, `ultralytics`, `mediapipe`, `opencv-python`, `numpy`)*

### Bước 3: Khởi chạy Web Application
```bash
python app.py
```

### Bước 4: Trải nghiệm trên trình duyệt
Truy cập: **[http://localhost:5000](http://localhost:5000)** (hoặc `http://127.0.0.1:5000`)
- **Dashboard (`/`):** Xem tổng quan đề tài và quy trình hoạt động.
- **Kết quả (`/results`):** Xem biểu đồ huấn luyện, ma trận nhầm lẫn và báo cáo thống kê.
- **Trải nghiệm (`/demo`):** Bấm nút **"Bắt đầu Camera"** để hệ thống nhận diện khuôn mặt và cảnh báo trực tiếp!
