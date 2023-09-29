# Video_Diffusion_Models
research paper: https://arxiv.org/pdf/2204.03458.pdf

## Kiến trúc mô hình: 
- Mô hình sử dụng kiến trúc 3D-Unet
- **Phần đầu vào:** Là một tensor 4 chiều với các trục được gắn nhãn là khung x chiều cao x chiều rộng x kênh, được xử lý theo cách phân tích theo không gian và thời gian như mô tả trong Phần 3. Đầu vào là một video nhiễu z, điều kiện c, và log SNR A.
- **Phần đầu ra:** Là một tensor 4 chiều với cùng kích thước và số kênh như đầu vào, nhưng không có nhiễu.

**Các ưu điểm của mô hình** bao gồm:

- Có thể tái tạo video một cách chính xác, ngay cả khi video bị nhiễu nhiều.
- Có thể tạo video mới từ điều kiện đầu vào.
- Có thể được sử dụng cho các ứng dụng như tạo video chuyển động, tạo video từ văn bản, và tạo video từ các mô hình 3D.

**Các hạn chế của mô hình** bao gồm:

- Cần nhiều dữ liệu để đào tạo mô hình.
- Mô hình có thể mất nhiều thời gian để training rất lâu so với **TGANv2**
![image](/assert/model.png)

## Bộ dữ liệu sử dụng **Moving MNIST**
- Bộ dữ liệu bao gồm 20 file gif
  
![example](/assert/sequence_0.gif)

## Kết quả sau 1530 epochs
![result](/assert/153.gif)
- Do độ phức tạp của U-net 3D và thời gian huấn luyện lâu. Nên hình ảnh sinh ra vẫn còn mơ hồ, chưa rõ ràng.
