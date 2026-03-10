# Phân tích **8 kiến trúc** có trong notebook (`Untitled-3.ipynb`) — cách hoạt động & đánh giá từng cái

> **Lưu ý quan trọng (từ code):** tất cả các mô hình trong notebook đều tạo **một mask phổ tĩnh cho cả đoạn** (model trả về shape `[B, 161]` qua `self.head` → `sigmoid`), rồi áp mask đó cho mọi frame: `enhanced = stft.inverse(spec * mask.unsqueeze(-1))`.  
> Nghĩa là các mô hình **không** dự đoán mask khung-đến-khung (time-varying) mà dùng **một mask chung cho toàn bộ clip** — điều này ảnh hưởng lớn tới hiệu năng trên nhiễu phi-định cư / non-stationary.

---

## Tổng nguồn gốc thành phần (dẫn xuất từ code)
- **Encoder options**:
  - `MobileOneEncoder` — dùng `timm.create_model("mobileone_s0", pretrained=True)`, lấy `backbone.num_features` làm `out_dim`. (feature lớn, pretrained CNN).
  - `DFNet1Encoder` — `ConvBlock` stack: Conv2d(1→16) → Conv2d(16→32) → Conv2d(32→64). `out_dim = 64`.
  - `DFNet2Encoder` — Conv stack: 1→32→64→128. `out_dim = 128`.
  - `DFNet3Encoder` — Conv stack: 1→32→64→128→256. `out_dim = 256`.
- **Temporal module**:
  - `TemporalGRU(dim)` — `nn.GRU(dim, dim, batch_first=True)`; forward trả về `h` cuối → shape `(B, dim)`.
  - `TemporalTCN(dim)` — causal-like dilated `Conv1d` stack:
    - `Conv1d(dim,dim,3,padding=2,dilation=2)` → `ReLU` → `Conv1d(..., padding=4,dilation=4)` → ... → cuối `mean(-1)` → `(B, dim)`.
    - **Ghi chú:** code dùng symmetric `padding` (padding > 0) — điều này *có thể* làm TCN **tham chiếu tới future frames** nếu không cắt-trừ, nên TCN như viết có khả năng **không hoàn toàn causal** cho streaming.
- **Head**: `nn.Linear(dim, 161)` → `sigmoid` → mask (161 bins).
- **Combos**: `configs = [("mobileone","gru"),("mobileone","tcn"),("dfnet1","gru"),("dfnet1","tcn"),("dfnet2","gru"),("dfnet2","tcn"),("dfnet3","gru"),("dfnet3","tcn")]` — tức 8 kiến trúc.

---

## Cách đọc đánh giá dưới đây
Mỗi mục mô tả **cách hoạt động ngắn** (data flow) → **điểm mạnh / điểm yếu** → **ưu tiên sử dụng** (use-case). Đánh giá dựa trên: cấu trúc code, kích thước `out_dim`, thực tế mask là *global per-utterance*, và tính khả dụng cho realtime/streaming.

---

### 1) `MobileOneEncoder + TemporalGRU`
**Cách hoạt động**
- Input: STFT complex → `magnitude(spec)` → encoder: chuyển magnitude thành 2D image-like, lặp 3 channel, chạy `mobileone_s0` backbone (pretrained) → cho feature vector lớn (`backbone.num_features`) → temporal GRU nhận sequence features (encoder thiết kế ở đây trả về feature lặp trên thời gian) → GRU trả `h_last` → linear → sigmoid → mask 161-bin áp cho mọi frame.

**Đánh giá**
- **Ưu:** Tiềm năng chất lượng cao nhờ backbone pretrained mạnh (trích đặc trưng phong phú). 
- **Nhược:** Rất nặng về compute và memory (MobileOne backbone lớn), inference chậm trên CPU/mobile; encoder chu trình xử lý toàn ảnh spectrogram (chi phí lớn).
- **Streaming:** KHÓ khăn — backbone xử lý toàn spectrogram & GRU dùng toàn sequence; không phù hợp realtime strict.
- **Use-case:** Khi bạn có GPU/desktop hoặc muốn exploit transfer learning, xử lý offline/nearline, hoặc thử benchmark chất lượng tối đa.

---

### 2) `MobileOneEncoder + TemporalTCN`
**Cách hoạt động**
- Giống trên nhưng temporal module là TCN (dilated Conv1d stack) → trả ra vector summary.

**Đánh giá**
- **Ưu:** TCN có receptive field lớn, song song hơn GRU trên GPU; có khả năng nắm long-range patterns tốt.
- **Nhược:** Do `padding` trong code, TCN có thể **dùng thông tin tương lai** (không causal) → không an toàn cho streaming; latency cao vì backbone vẫn nặng.
- **Streaming:** Không phù hợp (cả vì MobileOne lẫn khả năng non-causal của TCN).
- **Use-case:** Offline experiment để kiểm tra lợi ích receptive-field so với GRU trên feature-rich encoder.

---

### 3) `DFNet1Encoder + TemporalGRU`
**Cách hoạt động**
- `DFNet1Encoder`: nhẹ — 3 ConvBlock (1→16→32→64) → output dim=64; temporal GRU (dim=64) → `Linear(64,161)` → mask chung.

**Đánh giá**
- **Ưu:** Rất nhẹ, ít params và MACs → inference nhanh trên CPU/mobile; dễ train.
- **Nhược:** Biểu diễn hạn chế (out_dim=64) → giới hạn khả năng tách nhiễu phức tạp; do mask là global nên kém với nhiễu biến đổi.
- **Streaming:** Dễ apply hơn MobileOne; nhưng vì model vẫn dùng toàn sequence để lấy `h_last`, cần xử lý đoạn/chuỗi để streaming thực sự.
- **Use-case:** Thiết bị nhúng, prototype nhanh, khi budget strict và target là nhiễu tĩnh hoặc quasi-stationary.

---

### 4) `DFNet1Encoder + TemporalTCN`
**Cách hoạt động**
- DFNet1 nhẹ + TCN (dilated convs) → vector summary → mask.

**Đánh giá**
- **Ưu:** TCN có thể capture patterns dài hơn so với 1-layer GRU cùng dim; khi chạy trên GPU có thể nhanh hơn GRU.
- **Nhược:** TCN như hiện tại có thể non-causal; model vẫn hạn chế bởi encoder nhỏ.
- **Streaming:** Phải kiểm tra tính causal; nếu cần streaming, phải sửa padding/crop.
- **Use-case:** Khi muốn thêm receptive field mà vẫn hạn chế params.

---

### 5) `DFNet2Encoder + TemporalGRU`
**Cách hoạt động**
- DFNet2: 3 conv blocks với out_dim=128 (1→32→64→128) → GRU(dim=128) → head → mask.

**Đánh giá**
- **Ưu:** Cân bằng tốt hơn giữa biểu diễn (128 dim) và chi phí; kỳ vọng chất lượng tốt hơn DFNet1, đặc biệt với noise ít stationary.
- **Nhược:** Vẫn trả mask per-utterance → giới hạn với non-stationary noise.
- **Streaming:** Khả thi trên mobile trung bình nếu chia đoạn; GRU sequential cost nhưng nhẹ hơn MobileOne.
- **Use-case:** Mobile devices mid-range, khi muốn trade-off chất lượng/RTF tốt.

---

### 6) `DFNet2Encoder + TemporalTCN`
**Cách hoạt động**
- DFNet2 + TCN (dilated convs) → vector → mask.

**Đánh giá**
- **Ưu:** Có thể nắm bối cảnh dài hơn DFNet2+GRU; tiềm năng cải thiện phần transient / non-stationary trong đoạn (nếu TCN thiết kế causal).
- **Nhược:** Cần kiểm tra causal/padding; đôi khi TCN sâu có artifacts nếu không điều chỉnh.
- **Streaming:** Nếu sửa để causal, có thể phù hợp; hiện tại cần review padding behavior.
- **Use-case:** Khi muốn receptive field lớn hơn cho cùng budget encoder.

---

### 7) `DFNet3Encoder + TemporalGRU`
**Cách hoạt động**
- DFNet3: dày nhất trong DFNet (1→32→64→128→256), `out_dim = 256`; GRU(dim=256) → head → mask.

**Đánh giá**
- **Ưu:** Mạnh nhất về tính biểu diễn trong nhóm DFNet — kỳ vọng chất lượng tốt nhất cho các model Conv-based. Phù hợp khi muốn nâng chất lượng mà vẫn tránh backbone lớn.
- **Nhược:** Params và MACs tăng (so với DFNet1/2); vẫn tạo mask tĩnh → hạn chế non-stationary noise handling.
- **Streaming:** Nặng hơn DFNet2 nhưng vẫn nhẹ hơn MobileOne; cần chia đoạn nếu streaming.
- **Use-case:** Desktop nhẹ hoặc mobile cao cấp; khi muốn chất lượng speech enhancement tối ưu trong nhóm conv-encoder.

---

### 8) `DFNet3Encoder + TemporalTCN`
**Cách hoạt động**
- DFNet3 (out_dim 256) + TCN dilation stack → vector summary → head → mask.

**Đánh giá**
- **Ưu:** Kết hợp encoder mạnh + temporal receptive-field lớn → tiềm năng tốt nhất trong các biến thể DFNet để xử lý nhiễu phức tạp, transient và cấu trúc thời gian dài.
- **Nhược:** Tốn compute nhất trong nhóm DFNet; nếu TCN non-causal thì không dùng cho streaming real-time; nếu điều chỉnh causal, latency/receptive trade-offs cần cân bằng.
- **Streaming:** Khó nếu giữ cấu trúc hiện tại; khả thi với sửa causal & phân đoạn inference.
- **Use-case:** Khi mục tiêu là chất lượng tối đa trong phạm vi models do notebook định nghĩa, nhưng vẫn tránh MobileOne.

---

## Kết luận tóm tắt (practical takeaways)
1. **Tất cả 8 kiến trúc** trong notebook đều tạo **một mask phổ tĩnh cho toàn đoạn** → phù hợp với **nhiễu tĩnh / quasi-stationary** nhưng **không tốt** cho nhiễu thay đổi nhanh (bus, crowd, intermittent noises). Nếu mục tiêu của bạn là real-world non-stationary noise, cần sửa model để sinh **mask theo frame** (time-varying) hoặc output sequence masks.
2. **MobileOne variants** hứa hẹn đặc trưng giàu, nhưng **rất nặng** và không phù hợp realtime trên CPU/mobile nhỏ. Dùng cho offline / GPU.
3. **DFNet variants (1→3)** cung cấp dải trade-off rõ rệt:
   - DFNet1: rất nhẹ, phù hợp embedded.
   - DFNet2: balanced.
   - DFNet3: tốt nhất chất lượng trong nhóm conv-based.
4. **GRU vs TCN**:
   - GRU: đơn giản, sequential, dễ kiểm soát cho streaming nếu thiết kế frame-online; nhưng hiện code lấy toàn sequence → vẫn offline-like.
   - TCN: receptive field lớn, tốt cho pattern dài, nhưng code hiện tại có padding -> *cần kiểm tra/đổi cho causal* nếu streaming cần.
5. **Nếu muốn deploy realtime**:  
   - Bắt buộc: (A) thay head trả mask theo khung (shape `[B, T, F]`) thay vì global mask, (B) sửa TCN padding để causal hoặc dùng causal convolutions, (C) tránh MobileOne backbone hoặc dùng aggressive pruning/quantization.
6. **Nếu muốn so sánh trong notebook (offline)**: thứ tự chất lượng ước tính (thực nghiệm cần đo):  
   `MobileOne + (TCN/GRU)` ≈ tốt nhất (nếu có compute), tiếp theo `DFNet3+(TCN/GRU)` → `DFNet2+(TCN/GRU)` → `DFNet1+(TCN/GRU)` → với TCN hơi thường có lợi nếu receptive-field hữu ích.

---

Nếu bạn muốn, mình sẽ **(A)** tự động sinh 1 bảng Markdown so sánh 8 dòng (Quality / Params (relative) / Latency (relative) / Streaming-safe?) hoặc **(B)** chỉnh code notebook để output **mask per-frame** thay vì global mask và sửa TCN thành causal — làm xong và trả lại patch diff/ cell code ngay trong block code markdown. Bạn chọn A hoặc B (hoặc cả 2).