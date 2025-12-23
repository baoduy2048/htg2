# Speed Layer – Xử lý dữ liệu chứng khoán Real-time

Speed Layer chịu trách nhiệm thu thập và xử lý dữ liệu chứng khoán theo thời gian thực (Real-time), đảm bảo độ trễ thấp nhất để phục vụ các dashboard và hệ thống cảnh báo

## Giới thiệu

Hệ thống Speed Layer bao gồm 2 module chính hoạt động độc lập hoặc phối hợp qua Message Queue (Kafka):

- Crawler Speed (crawler_speed): Thu thập dữ liệu tick-by-tick (khớp lệnh, đặt lệnh) từ thị trường thông qua vnstock3 hoặc WebSocket và đẩy vào Kafka.

- OHLCV Processor (ohlcv): Lắng nghe dữ liệu từ Kafka, tổng hợp thành nến (Open, High, Low, Close, Volume) theo các khung thời gian ngay lập tức.

## Cấu trúc thư mục

speed/
├── crawler_speed/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── config.py
│   └── (source code: main.py, market_crawler.py, kafka_producer.py...)
└── ohlcv/
    ├── Dockerfile
    ├── requirements.txt
    └── (source code: ohlcv_speed.py...)

## Hướng dẫn sử dụng

```bash
git clone <repository-url>
cd speed

2️⃣ Tạo virtual environment (khuyến nghị)

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate     # Windows

3️⃣ Cài đặt thư viện Python

# Cho Crawler
cd crawler_speed
pip install -r requirements.txt

# Cho OHLCV
cd ../ohlcv
pip install -r requirements.txt
```
## Chạy Speed Layer với Docker

### 1. Crawler Speed (Thu thập dữ liệu)
```bash
cd speed/crawler_speed
docker build -t bigdata/speed-crawler:latest .
docker run -d \
  --name speed-crawler \
  --network host \
  -v /home/anh/speed_logs:/app/logs \
  bigdata/speed-crawler:latest python main.py
```
### 2. OHLCV Processor
```bash
cd speed/ohlcv
docker build -t bigdata/speed-ohlcv:latest .
docker run -d \
  --name speed-ohlcv \
  --network host \
  -v /home/anh/speed_logs:/app/logs \
  bigdata/speed-ohlcv:latest python ohlcv_speed.py
```
### 3. Debug interactive container
```bash
# Debug Crawler
docker run -it \
  --network host \
  -v $(pwd)/crawler_speed:/app \
  bigdata/speed-crawler:latest bash

# Trong container chạy thử:
# python main.py
```
