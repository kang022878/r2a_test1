# 프로젝트 이름
Ghibli Camera 3D

# 프로젝트 설명
한 줄 소개: 카메라로 비춘 피사체를 AI로 스타일링하고 즉시 3D 에셋으로 변환하는 실시간 웹 앱.
슬로건: Point it. Restyle it. Export it.

## 사용된 기술
- Backend: FastAPI, Python, OpenCV
- AI: YOLOv8 기반 객체 탐지/세그멘테이션, 스타일링 엔진(ghibli/pixel/toon), 3D 변환 파이프라인
- Frontend: Vite + React + TypeScript + Tailwind CSS
- 3D Preview: three.js (GLTFLoader/OrbitControls)

## 페이지별 기능
- `/` 랜딩 페이지
  - 제품 소개 및 데모 UI
- `/camera` 실시간 카메라
  - 카메라 입력
  - 객체 탐지 후 선택
  - 스타일 적용(ghibli/pixel/toon)
  - 3D Export → 프리뷰 이동
- `/preview?id=...` 3D 프리뷰
  - GLB 미리보기
  - 다운로드

## 로컬 실행
1. 백엔드 설치 및 실행
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 80
```

2. 프런트엔드 빌드
```bash
cd web
npm install
npm run build
```

## 참고
- 프런트엔드 빌드 결과는 `web/dist`에 생성되며, FastAPI가 이를 정적 파일로 서빙합니다.
- 정적 이미지 파일은 `web/public`에 두고, 경로는 `/image1.png`처럼 사용합니다.
