#!/bin/bash
# 같은 Wi-Fi의 폰에서 접속하려면 이렇게 실행하세요.
cd "$(dirname "$0")"
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 80
