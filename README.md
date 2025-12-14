
# Hashiwokakero (Bridges) Solver 🌉

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

Đồ án môn học **Cơ sở Trí tuệ Nhân tạo (Introduction to AI)**.
Dự án này triển khai các thuật toán tìm kiếm và suy diễn logic để giải quyết trò chơi đố trí **Hashiwokakero** (hay còn gọi là Hashi/Bridges).

## 📋 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Live demo](#-live-demo)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Các thuật toán](#-các-thuật-toán)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)

## 📖 Giới thiệu

**Hashiwokakero** là một trò chơi logic được chơi trên lưới hình chữ nhật. Mục tiêu là kết nối tất cả các đảo (số) bằng các cây cầu sao cho:
1. Số cầu nối với mỗi đảo bằng đúng số ghi trên đảo đó.
2. Các cầu chỉ đi ngang hoặc dọc, không cắt nhau.
3. Tối đa 2 cầu song song giữa hai đảo.
4. Tất cả các đảo phải tạo thành một đồ thị liên thông.

Dự án này giải quyết bài toán bằng cách mô hình hóa nó dưới dạng **CNF (Conjunctive Normal Form)** để giải bằng SAT Solver, đồng thời so sánh với các thuật toán tìm kiếm truyền thống như **A*** và **Backtracking**.

## 🌐 Live Demo

Trải nghiệm ngay ứng dụng trực tuyến (không cần cài đặt) tại:

👉 **[https://hashiwokakero.streamlit.app/](https://hashiwokakero.streamlit.app/)**

## 📂 Cấu trúc dự án

Dự án được cấu trúc thành các module Python riêng biệt để dễ bảo trì và mở rộng:

| File/Folder | Mô tả |
|-------------|-------|
| `app.py` | Ứng dụng Web giao diện trực quan (xây dựng bằng **Streamlit**). |
| `main.py` | Điểm khởi chạy chính (Entry point) cho CLI. |
| `model.py` | Định nghĩa lớp `HashiPuzzle` và xử lý dữ liệu đầu vào. |
| `logic.py` | Bộ sinh mệnh đề CNF (`CNFGenerator`) và các quy tắc logic. |
| `solvers.py` | Chứa cài đặt của tất cả thuật toán (PySAT, A*, Backtracking...). |
| `experiments.py`| Kịch bản chạy thực nghiệm, đánh giá hiệu năng và so sánh. |
| `utils.py` | Các hàm tiện ích (kiểm tra liên thông, vẽ đồ thị, xử lý file). |
| `inputs/` | Thư mục chứa các file input mẫu (`input-xx.txt`). |
| `outputs/` | Thư mục chứa kết quả giải và file CSV thống kê. |
| `requirements.txt`| Danh sách các thư viện cần thiết. |

## 🧠 Các thuật toán

Dự án triển khai và so sánh 5 phương pháp giải quyết vấn đề:

1.  **PySAT Solver (Glucose3):**
    *   Mô hình hóa bài toán thành các mệnh đề logic CNF.
    *   Sử dụng chiến lược lai: Giải SAT cho ràng buộc cục bộ + Kiểm tra đồ thị cho ràng buộc liên thông.
    *   *Hiệu năng:* 🚀 Nhanh nhất, giải được map 40x40 (500 đảo) < 0.5s.

2.  **A∗ Search (Advanced):**
    *   Sử dụng hàm Heuristic phức hợp.
    *   *Hiệu năng:* Tốt cho các map cỡ trung bình (< 100 đảo).

3.  **Optimized Backtracking (CSP):**
    *   Áp dụng lan truyền ràng buộc (Constraint Propagation).
    *   Kỹ thuật chọn biến MRV (Minimum Remaining Values) và LCV.
    *   *Hiệu năng:* Rất ổn định, chỉ thua PySAT.

4.  **Naive Backtracking:** Quay lui cơ bản (dùng để so sánh).
5.  **Brute Force:** Vét cạn (dùng để so sánh baseline).

## ⚙️ Cài đặt

Yêu cầu **Python 3.7** trở lên.

1. Clone repository:
   ```bash
   git clone https://github.com/RamanujanJR/Hashiwokakero.git
   ```

2. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
   *Các thư viện chính: `python-sat`, `numpy`, `pandas`, `matplotlib`, `streamlit`.*

## 🚀 Hướng dẫn sử dụng

### 1. Chạy ứng dụng Web (GUI)
Để trải nghiệm trực quan, xem lời giải và biểu đồ:
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`.

### 2. Chạy thực nghiệm (Benchmark)
Để chạy lại toàn bộ quá trình so sánh hiệu năng các thuật toán:
```bash
python main.py
```
Kết quả sẽ được lưu vào file CSV trong thư mục `outputs/`.