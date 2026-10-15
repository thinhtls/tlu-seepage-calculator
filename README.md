# Tính toán thấm công trình thủy lợi

Ứng dụng Streamlit tính thấm theo hai phương pháp Chugaev và Bligh–Lane, vẽ biểu đồ áp lực đẩy ngược và tải thuyết minh Markdown.

## Chạy trên máy

Dùng Python 3.12:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Triển khai Streamlit Community Cloud

1. Đưa nội dung thư mục này lên repository `tlu-seepage-calculator`, nhánh `main`.
2. Truy cập https://share.streamlit.io/ và chọn **Create app** → triển khai từ GitHub.
3. Chọn repository, branch `main`, main file path `app.py`.
4. Trong **Advanced settings**, chọn Python **3.12**.
5. Chọn tên miền `tinh-tham-thuy-cong.streamlit.app` nếu còn khả dụng, rồi nhấn **Deploy**.

Streamlit sẽ cài các thư viện trong `requirements.txt`. Thời gian triển khai phụ thuộc quá trình cài đặt; không cố định 60 giây. Các commit mới trên nhánh triển khai sẽ được Streamlit cập nhật sau khi xử lý xong.

Tài liệu chính thức: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Thay đổi so với mã gốc

- Sửa ký tự escape `\\xi_i` trong chuỗi thuyết minh để Python đọc được mã nguồn.
- Thêm khai báo phiên bản thư viện và cấu hình giao diện Streamlit.
- Cập nhật tùy chọn độ rộng bảng và cố định các mốc trục biểu đồ để tránh cảnh báo thư viện.
- Giữ nguyên các công thức tính toán và dữ liệu mẫu của mã gốc.

Việc kiểm tra triển khai xác nhận ứng dụng chạy được; không thay thế thẩm định các công thức và số liệu chuyên ngành.
