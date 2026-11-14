# Báo cáo cá nhân — Day 6

Bùi Đức Tuấn · 2A202603902 · L3B

## Dữ liệu và pipeline

- Chủ đề, nguồn/quyền, số trang và quy tắc tính trang: Corpus là sách OpenStax *Principles of Economics 2e* (Rice University, 2017), giấy phép CC BY 4.0, PDF 974 trang, truy cập 2026-11-14. Tôi học quản trị kinh doanh nên đây là sách nhập môn kinh tế quen thuộc, và ba chương cung cầu, thị trường lao động/tài chính, độ co giãn là phần tôi dùng nhiều nhất. Trang được tính (số trang PDF): chương 3 *Demand and Supply* 55-84 (30 trang), chương 4 *Labor and Financial Markets* 93-112 (20), chương 5 *Elasticity* 117-136 (20). Tổng 70 trang nội dung, 192.103 ký tự sau chuẩn hóa. Tôi bỏ Key Terms, Key Concepts and Summary, câu hỏi ôn tập và bài tập cuối chương, vì các phần này lặp lại định nghĩa trong bài và có thể "lộ" sẵn đáp án, làm truy xuất trúng phần tóm tắt thay vì đoạn giải thích gốc. Trang in = trang PDF − 10. Nguồn và danh sách trang ghi trong [`data/manifest.json`](../../data/manifest.json).
- Chuẩn hóa, lỗi và giới hạn: Đọc bằng PyMuPDF 2.1 theo text block (mỗi block gần bằng một đoạn). Chuẩn hóa Unicode NFKC, nối từ bị gạch nối cuối dòng, gộp khoảng trắng trong đoạn, giữ ranh giới đoạn, bỏ header/footer lặp (`Chapter N | …`, `This OpenStax book is available…`). Log ghi 2 trang ngắn là trang cuối chương (p112: 148 ký tự, p136: 315 ký tự), không có trang đọc lỗi ([`ingestion_log.json`](../../starter/results/ingestion_log.json)). Giới hạn lớn nhất là bảng: bảng số liệu bị tách thành từng ô (Table 3.3 trang 61 thành các dòng như `$2.00 460 700`, mất nhãn cột). Bảng vốn đọc theo hàng và cột, tách ra thì khó biết số nào thuộc cột nào, nên câu hỏi số liệu khó kiểm đúng. Hình chỉ còn chú thích.
- Embedding, vector store, lệnh chạy, chi phí: OpenAI `text-embedding-3-small`, 1536 chiều, cùng model cho document và query, không prefix. Chroma 1.5.9 `PersistentClient` tại `starter/chroma_db`, metric cosine, chỉ mục HNSW (tìm gần đúng), collection `strategy_A` và `strategy_B`. Lệnh (trong `starter/`): `python -m src.run verify-benchmark | ingest | evaluate | audit-sheets`. Chi phí embedding mỗi lần ingest: A 44.289 token ≈ $0,00089, B 40.233 token ≈ $0,00080 (giá $0,02/1M token); 40 câu hỏi gần như không tốn gì. Thư viện: pymupdf 2.1, chromadb 1.5.9, openai 3.51.0, tiktoken 0.14.0, streamlit 1.67.0, Python 3.13.
- Metadata, trường lọc: Mỗi chunk có `document_id`, `source`, `page`, `page_end`, `chapter` (int), `chapter_title`, `section` (vd `3.1`, `3.intro`), `strategy`; A có thêm `char_start`. Trường lọc là `chapter` (3/4/5), lấy từ trang mở chương và footer từng trang. Tôi chọn `chapter` vì mỗi chương là một mảng kiến thức riêng, người học thường biết mình đang hỏi về chương nào. Hạn chế: một chương có thể gộp nhiều chủ đề (chương 4 có cả lao động và tài chính), và một khái niệm có thể nằm ở nhiều chương (inferior good có ở chương 3 và chương 5), nên lọc một chương có thể làm mất bức tranh tổng thể. Ngoài ra `section` gắn theo đầu chunk, chunk chứa tiêu đề mục mới ở giữa vẫn mang mục cũ.
- Công cụ AI: Tôi dùng Claude Code để tìm và đánh giá nguồn tài liệu, đánh giá rủi ro, viết code pipeline và demo, hướng dẫn phần embedding, đề xuất câu benchmark (tôi duyệt từng câu) và hỗ trợ soạn báo cáo. Tôi tự kiểm bằng `verify-benchmark` (40/40 evidence khớp trang gold), chạy ingest hai lần để kiểm không trùng chunk, đối chiếu trace với trang sách gốc trên demo, và duyệt điểm chấm tay ở phần audit.

## Cấu hình và benchmark

- A / B / C:

  | | Chunk | Truy xuất | Số chunk |
  |---|---|---|---|
  | A | cố định 500 ký tự, overlap 50, trong từng trang | không lọc | 449 |
  | B | gom đoạn tối đa 1000 ký tự, mục mới mở chunk mới, đoạn dài tách ở cuối câu, được qua trang | không lọc | 260 |
  | C | dùng chung collection của B | lọc `chapter` khi câu hỏi có filter (7/40 câu), còn lại như B | chung B |

  A-B chỉ khác cách chunk; B-C chỉ khác filter (cùng chunk, model, câu hỏi). A giữ trong từng trang vì sách học thuật hay chuyển ý theo trang và đây là baseline đơn giản, có `char_start` để định vị. B cho qua trang vì đoạn văn hay đứt giữa hai trang PDF, cắt ở ranh giới trang sẽ cắt đôi evidence. Vì vậy A và B khác nhau cùng lúc ở ba thứ (cỡ chunk, cắt theo ký tự hay theo đoạn, qua trang hay không), nên so sánh A-B chưa tách được yếu tố nào tạo ra chênh lệch. Chunk ID = sha1(strategy|document|page|index|text); lần ingest thứ hai upsert A 449→449, B 260→260, xóa 0.
- Benchmark: 40 câu ([`benchmark/questions.json`](../../benchmark/questions.json)), chia trước khi chạy 29 dev / 11 test. Theo chương gold: ch3 13, ch4 15, ch5 12. Loại câu: direct 12, metadata_filter 7, condition 4, paraphrase 4, exception 2, cross_lingual 2, distractor 2, numeric 2, trap_misconception 1, counterintuitive 1, lay_language 1, cross_lingual_trap 1, multi_step 1. Bảy câu có filter: q14 (ch4), q15 (ch3), q29 (ch4), q40 (ch5) ở dev; t18 (ch4), t30 (ch5), t39 (ch4) ở test. Ba câu tiếng Việt: q21, q38, t34. Kiểm chứng evidence: `verify-benchmark` so evidence đã chuẩn hóa với văn bản trang gold, 40/40 đạt. Hit được tính khi chunk cùng tài liệu, trang gold nằm trong [page, page_end] và chunk chứa trọn evidence. Nguy cơ rò rỉ: q15 (dev) và t18 (test) cùng hỏi về price ceiling nhưng khác chương lọc; thực tế cả hai đều trượt ở A/B/C nên không thấy điểm test bị đẩy lên.
- Tiêu chí chọn chiến lược: ưu tiên dev Hit@3, sau đó coherence/grounding, cuối cùng là tài nguyên (số chunk, token, thời gian). Tiêu chí này chưa được ghi ra trước: lệnh `evaluate` in dev và test cùng lúc nên tôi đã thấy kết quả test khi chọn.

## Kết quả

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 11/29 | 12/29 | 449 | 6,05 s | 44.289 token ≈ $0,00089 |
| A | test | 8/11 | 9/11 | 449 | 6,05 s | 44.289 token ≈ $0,00089 |
| B | dev | 16/29 | 20/29 | 260 | 2,14 s | 40.233 token ≈ $0,00080 |
| B | test | 6/11 | 9/11 | 260 | 2,14 s | 40.233 token ≈ $0,00080 |
| C | dev | 16/29 | 20/29 | 260 (chung B) | 2,14 s | 40.233 token ≈ $0,00080 |
| C | test | 6/11 | 10/11 | 260 (chung B) | 2,14 s | 40.233 token ≈ $0,00080 |

Thời gian là lần ingest đầu (embed và upsert); lần hai A 4,19 s, B 2,38 s. Evidence bị tách qua chunk (trang gold có trong top-3 nhưng không chunk nào chứa trọn evidence): A dev 13 / test 1, B dev 3 / test 1, C dev 3 / test 0. Độ dài chunk: A median 500 ký tự, B median 784,5.

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | 0,9 (10) | 33/33 | 4/7 | 0/2/9 · 1,82 |
| B | 1,8 (10) | 33/33 | 2/7 | 0/2/9 · 1,82 |
| C | 1,8 (10, dùng chunk B) | 33/33 | 7/7 | 0/1/10 · 1,91 |

Filter tuân thủ là số câu có filter mà cả top-3 đều đúng chương; riêng test: A 2/3, B 1/3, C 3/3.

Trace, nhãn thủ công và cách chọn mẫu: Trace top-3 từng câu: [`traces_A.jsonl`](../../starter/results/traces_A.jsonl), [`traces_B.jsonl`](../../starter/results/traces_B.jsonl), [`traces_C.jsonl`](../../starter/results/traces_C.jsonl); metrics: [`metrics.csv`](../../starter/results/metrics.csv), [`filter_compliance.csv`](../../starter/results/filter_compliance.csv); cấu hình và kết quả: [`artifacts/experiment_log.csv`](../../artifacts/experiment_log.csv). Nhãn chấm tay nằm trong [`starter/results/audit/`](../../starter/results/audit/). Coherence: 10 vị trí chọn cố định trước khi chấm, là đoạn dài từ 200 ký tự trở lên đầu tiên của 10 trang đầu (p55-64); chấm chunk A và B chứa đoạn đó, C dùng lại điểm B. Grounding: 11 câu test, chấm top-3 có đủ căn cứ để trả lời không (0/1/2). Metadata: 11 câu test × top-3 = 33 bản ghi mỗi chiến lược, đối chiếu page, chapter, section.

## Phân tích và kết luận

- A-B: Dev Hit@3 tăng từ 12/29 lên 20/29 chủ yếu vì A cắt evidence nhiều: 13 ca ở dev so với 3 ca của B, vì cửa sổ 500 ký tự hay cắt giữa câu, giữa chữ. Coherence cùng chiều: A 0,9 so với B 1,8, chunk A thường mất câu kết hoặc dính mảnh bảng. Ở test thì ngược lại một chút: Hit@3 bằng nhau (9/11) còn Hit@1 A 8/11, B 6/11. A đứng hạng 1 ở t17, t20, t30 mà B không (B hạng 2 ở t17, t20 và trượt t30), B đứng hạng 1 ở t23 mà A không. Chunk A ngắn nên vector tập trung vào câu định nghĩa hơn. Test chỉ có 11 câu, mỗi câu là 9 điểm phần trăm, nên chênh 2 câu chưa đủ để kết luận B kém hơn A ở Hit@1.
- B-C: Filter làm top-3 đúng chương từ 2/7 lên 7/7, nhưng Hit@3 chỉ đổi ở một câu (t30: B trượt, C trúng hạng 2), dev giống hệt B. Filter giúp khi các đoạn ở chương khác gần nghĩa hơn evidence thật và đẩy nó ra khỏi top-3; nó không giúp khi evidence vẫn không vào được top-3 ngay trong đúng chương (q14, q15, q29, q40, t18 vẫn trượt). Lúc demo, coach hỏi về quan hệ giữa giá và cầu: không lọc thì kết quả lẫn sang chương 4, lọc chương 3 thì ra đúng phần cung cầu.
- Ví dụ sai 1, q01 "What is consumer surplus?" (dev, lỗi của A): A lấy đúng trang 81 nhưng câu định nghĩa bị cắt giữa chữ qua hai chunk ("…minus the a" / "iduals would…"), nên không chunk nào chứa trọn evidence: Hit@3 trượt; B trúng hạng 1. Nguyên nhân nằm ở chính cách cắt cố định 500 ký tự của A.
- Ví dụ sai 2, t30 "How can you tell a good is inferior?" (test, filter ch5, lỗi của B): A trúng hạng 1, B trượt, C trúng hạng 2. Top-3 của B lẫn trang 64-65 chương 3, nơi sách định nghĩa inferior good theo thu nhập, và không chunk nào chứa trọn vế "income elasticity âm" ở trang 133. Nguyên nhân là cùng khái niệm xuất hiện ở hai chương; lọc chương 5 loại được các đoạn chương 3.
- Chiến lược chọn: Tôi chọn B làm mặc định và dùng C khi người dùng cho biết chương. So với A, B ít chunk hơn (260 so với 449), ít token hơn (40.233 so với 44.289), ingest nhanh hơn (2,14 s so với 6,05 s), chunk dài hơn nên ít cắt evidence và đọc tự hiểu được hơn; chunk dài cũng hợp hơn khi đưa top-3 cho LLM trả lời. Đánh đổi là vector loãng hơn, Hit@1 test thấp hơn A (6/11 so với 8/11). Chi phí embedding đều dưới $0,001 nên tài nguyên không phải yếu tố quyết định. Nếu chủ yếu hỏi định nghĩa ngắn và cần top-1 chính xác, A vẫn đáng cân nhắc.
- Hạn chế: Test chỉ 11 câu, chênh một câu là 9 điểm phần trăm. Benchmark do tôi tự gán nhãn (có hỗ trợ của AI) nên có thể thiên lệch về cách diễn đạt của sách. Tôi đã thấy test cùng lúc với dev khi chọn cấu hình. Bảng bị tách ô nên câu số liệu phụ thuộc may rủi (t23 ở A trượt vì evidence bị tách). Chỉ có 3 câu tiếng Việt và điểm top-1 của chúng khoảng 0,3, so với 0,5-0,7 của câu tiếng Anh; q21 trượt ở cả ba. Điểm chấm tay chỉ có một người chấm. Thời gian đo một lần trên máy cá nhân, gồm cả độ trễ mạng gọi API, nên chỉ để so sánh tương đối; Chroma tìm gần đúng nên xếp hạng có thể lệch nhẹ.
- Nếu làm lại: Cần nhất là xử lý bảng: trích bảng nguyên khối, gắn tên cột vào từng hàng thay vì tách ô. Tiếp theo là chunk theo câu có overlap để không cắt evidence giữa chữ. Với câu tiếng Việt, dịch câu hỏi sang tiếng Anh trước khi tìm (thử trên demo, q21 sau khi dịch tìm ra trang 118 với điểm 0,65). Cuối cùng là ghi tiêu chí chọn trước và chỉ chạy test một lần, mở rộng test lên vài chục câu.
