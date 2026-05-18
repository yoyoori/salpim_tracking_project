# 🩺 살핌

고령자의 위험 행동을 실시간으로 인식하고 알림을 제공하는 AI 기반 행동 인식 시스템

---

# 📌 프로젝트 소개

살핌은 고령자의 일상 행동 중 위험 상황을 실시간으로 탐지하기 위한 컴퓨터 비전 기반 행동 인식 프로젝트이다.

ETRI-Activity3D 데이터셋 기반으로 위험 행동을 학습하며 RGB 영상과 Skeleton 데이터를 활용하여 행동 클래스를 분류한다.  
최종적으로 사용자가 영상을 입력하면 위험 행동 여부를 예측하고 결과를 웹 UI에서 확인할 수 있도록 구현하는 것을 목표로 한다.

---

# ✨ 주요 기능

- RGB 영상 기반 행동 인식
- Skeleton 관절 데이터 기반 행동 분류
- 위험 행동 실시간 탐지
- Streamlit 기반 웹 UI 제공
- FastAPI 기반 추론 서버 구축
- 행동 클래스별 Confidence Score 출력
- Bounding Box 및 Pose 시각화
- Threshold 조절 인터랙션 기능

---

# 📂 사용 데이터셋

## ETRI-Activity3D

고령자 행동 인식을 위해 구축된 3D 행동 데이터셋

- 100명의 참가자
- 55개 행동 클래스 제공
- RGB / Depth / Skeleton 데이터 포함

본 프로젝트에서는 위험 행동 중심의 10개 행동 클래스를 선택하여 사용한다.

---

# 🏷️ 행동 클래스

| 클래스 ID | 행동 코드 | 행동 설명 |
| --- | --- | --- |
| 0 | A010 | 침대에 눕기 |
| 1 | A011 | 침대에서 일어나기 |
| 2 | A016 | 의자에 앉기 |
| 3 | A018 | 의자에서 일어나기 |
| 4 | A023 | 바닥에 앉기 |
| 5 | A031 | 물건 줍기 |
| 6 | A035 | 넘어지기 |
| 7 | A041 | 비틀거리기 |
| 8 | A053 | 도움 요청하기 |
| 9 | A054 | 쓰러진 상태 유지 |

---

# 🗂️ 프로젝트 구조

```bash
salpim/
├── data/
│   ├── raw/
│   ├── processed/
│   └── skeleton/
├── models/
├── streamlit/
├── fastapi/
├── notebooks/
├── utils/
├── requirements.txt
└── README.md
```

---

## 전체 파이프라인

```text
입력 영상
→ 사람 검출
→ Pose 추출
→ Skeleton 생성
→ 행동 분류
→ 위험 행동 탐지
→ 알림 출력
```

---

# 🖥️ 웹 UI

## Streamlit

- 영상 업로드
- 추론 결과 시각화
- 행동 클래스 출력
- Confidence Score 출력
- Bounding Box 출력
- Threshold 실시간 조절

## FastAPI

- 모델 추론 API 제공
- Streamlit과 연동
- 결과 JSON 반환

---

# 🚀 실행 방법

## 1. 환경 설정

```bash
pip install -r requirements.txt
```

## 2. FastAPI 실행

```bash
uvicorn api:app --reload
```

## 3. Streamlit 실행

```bash
streamlit run app.py
```

---

# 🛠️ 기술 스택

- Python
- PyTorch
- YOLOv8
- MediaPipe
- OpenCV
- Streamlit
- FastAPI

---

# 👥 팀 소개

| 역할 | 담당 |
| --- | --- |
| 데이터 전처리 |  |
| Baseline 모델 구현 |  |
| 모델 고도화 |  |
| 웹 UI 개발 |  |
| 발표 및 문서화 |  |

---

# 📄 License

This project is for educational and research purposes.
