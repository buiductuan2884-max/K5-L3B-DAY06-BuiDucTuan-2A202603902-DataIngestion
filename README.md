# Day 06 · Data Ingestion Engineering

Kho tri thức truy xuất xây từ sách OpenStax *Principles of Economics 2e* (chương 3 đến 5, 70 trang nội dung, CC BY 4.0). Ba chiến lược dữ liệu được so sánh trên cùng corpus, cùng embedding model và cùng bộ 40 câu hỏi.

Đề bài gốc: [VinUni-AI20k/K5-L3-Day06-DataIngestionEngineering](https://github.com/VinUni-AI20k/K5-L3-Day06-DataIngestionEngineering) (xem `GUIDE.md`, `BENCHMARK.md`, `RUBRIC.md`).

| Thành phần | Lựa chọn |
|---|---|
| Đọc PDF | PyMuPDF 2.1 (text block ≈ đoạn văn), chuẩn hóa NFKC, bỏ header/footer |
| Embedding | OpenAI `text-embedding-3-small`, 1536 chiều, cùng model cho query và document |
| Vector store | Chroma 1.5.9 `PersistentClient` tại `starter/chroma_db`, cosine, HNSW (approximate) |
| Trường lọc | `chapter` (3, 4, 5) |

## Ba cấu hình

| | Chunk | Truy xuất |
|---|---|---|
| A | Cố định 500 ký tự, overlap 50, trong từng trang | Không lọc |
| B | Theo đoạn/mục, tối đa 1000 ký tự, có thể qua trang | Không lọc |
| C | Giống B (cùng collection) | Lọc `chapter` khi câu hỏi có filter |

## Cấu trúc

```text
data/manifest.json            nguồn, giấy phép, trang được tính
benchmark/questions.json      40 câu (29 dev / 11 test), gold page + evidence
artifacts/experiment_log.csv  cấu hình và kết quả A/B/C
starter/src/                  ingestion, chunking, embeddings, store, evaluation, run
starter/results/              chunk, trace top-3, metrics, filter compliance, bảng chấm tay (audit/)
starter/chroma_db/            vector store đã nạp (để chạy demo không cần nạp lại)
starter/app.py                demo Streamlit
report/individual/            báo cáo cá nhân
```

## Cài đặt

Python 3.11 đến 3.13. Chạy từ thư mục `starter/`:

```bash
python -m venv .venv
.venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # điền OPENAI_API_KEY
```

Tải PDF về `starter/data/Economics2e.pdf` từ URL trong `data/manifest.json` (chỉ cần khi chạy lại ingest).

## Chạy pipeline

```bash
python -m src.run verify-benchmark   # kiểm evidence nằm đúng trang gold, không gọi API
python -m src.run ingest             # chunk A/B, embed, upsert Chroma; chạy lần 2 để kiểm không trùng
python -m src.run evaluate           # top-3 cho A/B/C, ghi trace và metrics
python -m src.run audit-sheets       # tạo bảng chấm tay coherence / grounding / metadata
python scripts/cache_query_vectors.py  # lưu vector câu hỏi benchmark cho demo
```

Chi phí embedding mỗi lần ingest khoảng 84k token (< $0.002). Thời gian và token từng lần chạy ghi trong `starter/results/ingest_report.json`.

## Demo

```bash
streamlit run app.py
```

App có bốn tab. Kho tri thức cho xem nguồn, thống kê chunk, từng trang và các chunk của trang đó. Hỏi thử đặt top-3 của A, B, C cạnh nhau và cho lọc theo chương. Benchmark hiện metrics và trace từng câu. Trả lời (LLM) dùng `gpt-4o-mini` trả lời từ top-3 có trích trang, kèm tùy chọn dịch câu hỏi sang tiếng Anh trước khi tìm.

Deploy Streamlit Community Cloud: main file `starter/app.py`, thêm `OPENAI_API_KEY` trong Secrets.

## Ghi nguồn

Nội dung corpus: OpenStax, *Principles of Economics 2e*, Rice University, 2017, CC BY 4.0, https://openstax.org/details/books/principles-economics-2e. Văn bản được trích xuất và chuẩn hóa để thử nghiệm truy xuất.
