import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import tempfile
import base64
import io
import cv2
from collections import Counter
import uuid
import os
import base64

API_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_URL}/predict"
RESET_URL = f"{API_URL}/reset"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_PATH = os.path.join(BASE_DIR, "assets", "alarm.wav")

ACTION_MAP = {
    "A010": "양치하기", 
    "A011": "손 씻기", 
    "A016": "머리 빗기", 
    "A018": "상의 입기", 
    "A023": "진공 청소기 사용하기", 
    "A031": "리모컨으로 TV 컨트롤하기", 
    "A035": "전화 걸거나 받기", 
    "A041": "맨손 체조 하기", 
    "A053": "쓰러지기", 
    "A054": "누워있다 일어나기",
    "collecting": "프레임 수집 중",
    "normal": "정상 행동",
    "fall_down": "넘어짐",
}

RISK_LABELS = ["A053", "fall_down"]

st.set_page_config(
    page_title="위험 행동 인식 시스템",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .sub-text {
        font-size: 17px;
        color: #666;
        margin-bottom: 24px;
    }
    .info-card {
        padding: 20px;
        border-radius: 16px;
        background-color: #f8f9fb;
        border: 1px solid #e6e8ee;
        margin-bottom: 16px;
    }
    .danger-box {
        padding: 20px;
        border-radius: 16px;
        background-color: #fff1f1;
        border: 1px solid #ffcccc;
        color: #b42318;
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 18px;
    }
    .safe-box {
        padding: 20px;
        border-radius: 16px;
        background-color: #effaf3;
        border: 1px solid #c7eed8;
        color: #16783f;
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 18px;
    }
    .wait-box {
        padding: 20px;
        border-radius: 16px;
        background-color: #fff8e6;
        border: 1px solid #ffe0a3;
        color: #92400e;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 18px;
    }
    .result-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #ffffff;
        border: 1px solid #e6e8ee;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .stImage img {
        max-width: 700px !important;
        margin: auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">위험 행동 인식 시스템</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">영상 프레임을 FastAPI 모델 서버로 전송하고 RGB + Skeleton Fusion 모델의 행동 인식 결과를 확인합니다.</div>',
    unsafe_allow_html=True,
)


def get_font(size=36):
    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass

    return ImageFont.load_default()


def draw_predictions(image, predictions):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    font = get_font(34)

    SKELETON_EDGES = [
    # face
    (0, 13), (0, 14),
    (13, 15), (14, 16),

    # upper body
    (1, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),

    # torso
    (1, 7), (2, 8),
    (7, 8),

    # legs
    (7, 9), (9, 11),
    (8, 10), (10, 12),
]

    for pred in predictions:
        if "bbox" not in pred:
            continue

        x1, y1, x2, y2 = pred["bbox"]
        label = pred.get("label", "unknown")
        confidence = pred.get("confidence", 0.0)
        action_name = pred.get("action") or ACTION_MAP.get(label, "알 수 없는 행동")
        ready = pred.get("ready", True)
        sequence_count = pred.get("sequence_count", 0)
        sequence_required = pred.get("sequence_required", 90)
        is_risk = pred.get("is_risk", label in RISK_LABELS)

        track_id = pred.get("track_id", "-")
        if ready:
            text = f"ID {track_id} | {label} | {action_name} | {confidence:.2f}"
        else:
            text = f"ID {track_id} | collecting | {sequence_count}/{sequence_required}"

        color = "red" if is_risk else "green"

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=5,
        )
        
        keypoints = pred.get("keypoints", []) or []

        # 관절 연결선
        for a, b in SKELETON_EDGES:
            if a < len(keypoints) and b < len(keypoints):
                try:
                    x_a, y_a = keypoints[a]
                    x_b, y_b = keypoints[b]
                    draw.line(
                        [(x_a, y_a), (x_b, y_b)],
                        fill=color,
                        width=4,
                    )
                except Exception:
                    continue

        # MediaPipe keypoint를 원본 프레임 좌표에 점으로 표시
        for point in keypoints:
            try:
                px, py = point
                r = 4
                draw.ellipse((px - r, py - r, px + r, py + r), fill=color)
            except Exception:
                continue

        text_x = x1
        text_y = max(y1 - 44, 0)
        text_bbox = draw.textbbox((text_x, text_y), text, font=font)

        draw.rectangle(text_bbox, fill=color)
        draw.text((text_x, text_y), text, fill="white", font=font)

    return draw_image


def decode_b64_image(image_b64):
    if not image_b64:
        return None
    try:
        image_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    
def autoplay_alarm(audio_path):
    if not os.path.exists(audio_path):
        return 
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode()

    st.markdown(
        f"""
        <audio autoplay style="display:none;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def summarize_debug_entry(entry):
    track_id = entry.get("track_id")
    stage = entry.get("stage", "unknown")
    bbox = entry.get("bbox")
    crop_size = entry.get("crop_size", {})
    pose_detected = entry.get("pose_detected", False)
    sequence_count = entry.get("sequence_count", 0)
    sequence_required = entry.get("sequence_required", 90)
    label = entry.get("label", "-")
    confidence = entry.get("confidence", 0.0)

    return (
        f"Track {track_id} | {stage} | "
        f"bbox={bbox} | crop={crop_size.get('width', 0)}x{crop_size.get('height', 0)} | "
        f"pose={pose_detected} | seq={sequence_count}/{sequence_required} | "
        f"label={label} | conf={confidence:.2f}"
    )


def render_debug_frame(frame_debug):
    st.markdown(f"### Frame {frame_debug['frame_index']} / {frame_debug['timestamp']:.2f}초")

    original_image = frame_debug.get("image")
    all_predictions = frame_debug.get("all_predictions", [])
    if original_image is not None:
        st.image(
            draw_predictions(original_image, all_predictions),
            caption="1단계: 원본 프레임 + YOLO bbox + ByteTrack ID + 전역 keypoints",
            use_container_width=True,
        )

    debug_entries = frame_debug.get("debug_entries", [])
    if not debug_entries:
        st.info("이 프레임에는 검출된 사람/디버그 항목이 없습니다.")
        return

    for idx, entry in enumerate(debug_entries, start=1):
        with st.expander(f"{idx}. {summarize_debug_entry(entry)}", expanded=(idx == 1)):
            st.write(f"메시지: {entry.get('message', '-')}")
            st.write(f"원본 bbox: {entry.get('original_bbox')}")
            st.write(f"보정 bbox: {entry.get('bbox')}")
            st.write(f"crop 크기: {entry.get('crop_size')}")
            st.write(f"MediaPipe skeleton 검출 여부: {entry.get('pose_detected')}")
            st.write(f"현재 행동 예측: {entry.get('label', '-')} / {entry.get('action', '-')} / {entry.get('confidence', 0.0):.4f}")
            st.write(f"raw 예측: {entry.get('raw_label', '-')} / {entry.get('raw_action', '-')} / {entry.get('raw_confidence', 0.0):.4f}")

            col_a, col_b = st.columns(2)
            with col_a:
                crop_img = decode_b64_image(entry.get("crop_image_b64"))
                if crop_img is not None:
                    st.image(crop_img, caption="2단계: bbox crop 확인", use_container_width=True)
                else:
                    st.info("crop 이미지 없음")
            with col_b:
                skeleton_img = decode_b64_image(entry.get("crop_skeleton_image_b64"))
                if skeleton_img is not None:
                    st.image(skeleton_img, caption="3단계: crop 내부 skeleton 확인", use_container_width=True)
                else:
                    st.info("skeleton 이미지 없음")


def reset_api_session(session_id):
    response = requests.post(
        RESET_URL,
        data={"session_id": session_id},
        timeout=30,
    )
    response.raise_for_status()


def predict_image(image, session_id, debug=False):
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_img.name, format="JPEG")
    temp_img.close()

    with open(temp_img.name, "rb") as f:
        files = {"file": ("frame.jpg", f, "image/jpeg")}
        data = {"session_id": session_id, "debug": str(bool(debug)).lower()}
        response = requests.post(PREDICT_URL, files=files, data=data, timeout=120)
        response.raise_for_status()
        result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])

    return result


def extract_video_frames(uploaded_file, frame_stride):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        fps = 30

    duration_sec = total_frames / fps if total_frames > 0 else 0
    frame_stride = max(int(frame_stride), 1)

    frames = []
    frame_idx = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_idx % frame_stride == 0:
            timestamp = frame_idx / fps
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            frames.append({
                "timestamp": timestamp,
                "image": image,
            })

        frame_idx += 1

    cap.release()

    return frames, duration_sec, fps, total_frames


def filter_ready_predictions(predictions, confidence_threshold, show_collecting):
    filtered = []

    for pred in predictions:
        label = pred.get("label")
        ready = pred.get("ready", True)
        confidence = pred.get("confidence", 0.0)

        if not ready:
            if show_collecting:
                filtered.append(pred)
            continue

        if label == "collecting":
            if show_collecting:
                filtered.append(pred)
            continue

        if confidence >= confidence_threshold:
            filtered.append(pred)

    return filtered


def analyze_uploaded_file(uploaded_file, confidence_threshold, frame_stride, show_collecting, debug_mode=False, debug_sample_stride=30):
    file_type = uploaded_file.type
    session_id = str(uuid.uuid4())
    reset_api_session(session_id)

    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        api_result = predict_image(image, session_id, debug=debug_mode)
        predictions = api_result.get("predictions", [])
        filtered_predictions = filter_ready_predictions(
            predictions,
            confidence_threshold,
            show_collecting=True,
        )

        debug_frames = []
        if debug_mode:
            debug_frames.append({
                "frame_index": 0,
                "timestamp": 0.0,
                "image": image,
                "all_predictions": predictions,
                "debug_entries": api_result.get("debug", {}).get("entries", []),
            })

        return {
            "file_type": "image",
            "image": image,
            "predictions": filtered_predictions,
            "debug_frames": debug_frames,
            "session_id": session_id,
        }

    if "video" in file_type:
        frames, duration_sec, fps, total_frames = extract_video_frames(uploaded_file, frame_stride)

        frame_results = []
        debug_frames = []
        all_labels = []
        risk_events = []

        progress_bar = st.progress(0)
        status_area = st.empty()

        if not frames:
            raise ValueError("영상에서 프레임을 읽지 못했습니다.")

        for i, frame_data in enumerate(frames):
            timestamp = frame_data["timestamp"]
            image = frame_data["image"]

            api_result = predict_image(image, session_id, debug=debug_mode)
            predictions = api_result.get("predictions", [])
            filtered_predictions = filter_ready_predictions(
                predictions,
                confidence_threshold,
                show_collecting,
            )

            if debug_mode and (i % max(int(debug_sample_stride), 1) == 0):
                debug_frames.append({
                    "frame_index": i,
                    "timestamp": timestamp,
                    "image": image,
                    "all_predictions": predictions,
                    "debug_entries": api_result.get("debug", {}).get("entries", []),
                })

            for pred in filtered_predictions:
                label = pred.get("label")
                ready = pred.get("ready", True)

                if not ready or label == "collecting":
                    continue

                all_labels.append(label)

                if pred.get("is_risk", label in RISK_LABELS):
                    risk_events.append({
                        "timestamp": timestamp,
                        "label": label,
                        "action": pred.get("action") or ACTION_MAP.get(label, "알 수 없는 행동"),
                        "confidence": pred.get("confidence", 0.0),
                        "bbox": pred.get("bbox"),
                        "image": image,
                        "predictions": filtered_predictions,
                    })

            frame_results.append({
                "timestamp": timestamp,
                "image": image,
                "predictions": filtered_predictions,
            })

            progress_bar.progress((i + 1) / len(frames))
            status_area.info(f"분석 중: {i + 1}/{len(frames)} 프레임")

        return {
            "file_type": "video",
            "duration_sec": duration_sec,
            "fps": fps,
            "total_frames": total_frames,
            "analyzed_frames": len(frames),
            "frame_results": frame_results,
            "debug_frames": debug_frames,
            "label_counts": Counter(all_labels),
            "risk_events": risk_events,
            "session_id": session_id,
        }

    raise ValueError("지원하지 않는 파일 형식입니다.")


with st.sidebar:
    st.header("서버 상태")

    if st.button("FastAPI 연결 확인"):
        try:
            response = requests.get(API_URL, timeout=10)
            response.raise_for_status()
            st.success(response.json())
        except Exception as e:
            st.error(f"서버 연결 실패: {e}")

    st.caption("먼저 터미널에서 `uvicorn api:app --reload`를 실행해야 합니다.")


tab1, tab2, tab3, tab4 = st.tabs([
    "1. 업로드 및 설정",
    "2. 분석 진행",
    "3. 결과 확인",
    "4. 디버그 (개발자용)",
])

with tab1:
    st.subheader("파일 업로드 및 분석 설정")

    uploaded_file = st.file_uploader(
        "이미지 또는 영상 업로드",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

    with col2:
        frame_stride = st.slider(
            "영상 프레임 전송 간격",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
            help="1이면 모든 프레임을 서버로 전송합니다. Fusion 모델은 90프레임 누적 후 예측됩니다.",
        )

    with col3:
        show_collecting = st.checkbox(
            "수집 중 bbox 표시",
            value=True,
            help="90프레임이 쌓이기 전 collecting 상태의 bbox도 결과에 표시합니다.",
        )
    with col4:
        debug_mode = st.checkbox(
            "디버그 결과 수집",
            value=True,
            help="YOLO bbox, crop, skeleton, 행동 예측 단계를 4번 탭에서 확인합니다.",
        )
        debug_sample_stride = st.number_input(
            "디버그 저장 간격",
            min_value=1,
            max_value=300,
            value=30,
            step=1,
            help="영상 분석 시 몇 번째 분석 프레임마다 디버그 이미지를 저장할지 설정합니다.",
        )


    if uploaded_file is not None:
        st.markdown(
            f"""
            <div class="info-card">
            <b>업로드 파일명:</b> {uploaded_file.name}<br>
            <b>파일 타입:</b> {uploaded_file.type}<br>
            <b>Confidence threshold:</b> {confidence_threshold}<br>
            <b>영상 프레임 전송 간격:</b> {frame_stride}프레임마다 1장<br>
            <b>디버그 결과 수집:</b> {debug_mode}<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "image" in uploaded_file.type:
            preview_image = Image.open(uploaded_file).convert("RGB")
            st.image(preview_image, caption="업로드 이미지 미리보기", use_container_width=True)
            st.warning("현재 Fusion 모델은 90프레임 sequence 기반이라 단일 이미지는 collecting 상태까지만 확인될 수 있습니다.")

        elif "video" in uploaded_file.type:
            st.video(uploaded_file)

        st.info("설정을 확인한 뒤 2번 탭에서 분석을 시작하세요.")
    else:
        st.warning("먼저 이미지 또는 영상을 업로드하세요.")

with tab2:
    st.subheader("분석 진행")

    if uploaded_file is None:
        st.warning("1번 탭에서 먼저 파일을 업로드하세요.")
    else:
        st.markdown(
            """
            <div class="info-card">
            분석 시작 버튼을 누르면 FastAPI 서버로 프레임을 전송합니다.<br>
            영상은 같은 사람의 skeleton sequence가 90프레임 이상 쌓인 뒤부터 실제 행동 라벨을 예측합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("분석 시작", type="primary"):
            try:
                with st.spinner("분석 중입니다."):
                    result = analyze_uploaded_file(
                        uploaded_file,
                        confidence_threshold,
                        frame_stride,
                        show_collecting,
                        debug_mode,
                        debug_sample_stride,
                    )

                st.session_state["analysis_result"] = result

                # 위험 행동 감지 시 팝업 알림 상태 저장
                if result.get("file_type") == "video" and result.get("risk_events"):
                    first_risk = result["risk_events"][0]
                    st.session_state["risk_alert"] = {
                        "show": True,
                        "action": first_risk.get("action", "위험 행동"),
                        "timestamp": first_risk.get("timestamp", 0.0),
                        "confidence": first_risk.get("confidence", 0.0),
                    }
                else:
                    st.session_state["risk_alert"] = {"show": False}

                st.success("분석이 완료되었습니다. 3번 탭에서 결과를 확인하세요.")

            except requests.exceptions.ConnectionError:
                st.error("FastAPI 서버에 연결할 수 없습니다. `uvicorn api:app --reload`가 실행 중인지 확인하세요.")
            except Exception as e:
                st.error(f"오류 발생: {e}")

with tab3:
    st.subheader("결과 확인")

    if "analysis_result" not in st.session_state:
        st.warning("아직 분석 결과가 없습니다. 2번 탭에서 분석을 먼저 실행하세요.")

    else:
        result = st.session_state["analysis_result"]

        if result["file_type"] == "video":
            duration_sec = result["duration_sec"]
            frame_results = result["frame_results"]
            label_counts = result["label_counts"]
            risk_events = result["risk_events"]
            # 위험 행동 감지 팝업 + 알림음
            risk_alert = st.session_state.get("risk_alert", {})

            if risk_alert.get("show"):
                st.toast(
                    f"🚨 위험 행동 감지: {risk_alert['action']} / {risk_alert['timestamp']:.1f}초",
                    icon="🚨"
                )

                autoplay_alarm(ALARM_PATH)

                st.session_state["risk_alert"]["show"] = False

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("영상 길이", f"{duration_sec:.1f}초")

            with col2:
                st.metric("전체 프레임 수", result.get("total_frames", 0))

            with col3:
                st.metric("분석 프레임 수", result.get("analyzed_frames", len(frame_results)))

            if risk_events:
                first_risk = risk_events[0]
                st.markdown(
                    f"""
                    <div class="danger-box">
                    위험 행동 감지<br>
                    행동: {first_risk["action"]}<br>
                    최초 발생 시점: {first_risk["timestamp"]:.1f}초<br>
                    Confidence: {first_risk["confidence"]:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="safe-box">
                    위험 행동이 감지되지 않았습니다.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("행동별 등장 횟수")

            if label_counts:
                displayed = False
                for label in ACTION_MAP.keys():
                    count = label_counts.get(label, 0)
                    if count > 0:
                        action = ACTION_MAP.get(label, "알 수 없는 행동")
                        st.write(f"- {label} / {action}: {count}회")
                        displayed = True
                if not displayed:
                    st.write("아직 행동 결과가 없습니다.")
            else:
                st.write("아직 실제 행동 라벨로 확정된 결과가 없습니다. 영상이 너무 짧거나 프레임 전송 간격이 클 수 있습니다.")

            st.subheader("위험 감지 장면")

            if risk_events:
                risk_scene = risk_events[0]

                risk_image = risk_scene["image"]
                risk_predictions = risk_scene["predictions"]
                risk_draw_image = draw_predictions(risk_image, risk_predictions)

                st.write(f"위험 행동 감지 시점: {risk_scene['timestamp']:.1f}초")
                st.image(risk_draw_image, use_container_width=True)

            else:
                st.info("위험 행동이 감지되지 않아 표시할 위험 감지 장면이 없습니다.")

            st.subheader("타임라인 결과")

            timeline_events = []
            previous_label = None

            for frame_result in frame_results:
                timestamp = frame_result["timestamp"]
                predictions = frame_result["predictions"]

                if not predictions:
                    continue

                # confidence가 가장 높은 예측 1개 선택
                top_pred = predictions[0]

                label = top_pred.get("label", "unknown")

                # collecting이나 unknown은 타임라인에서 제외
                if label in ["collecting", "unknown"]:
                    continue

                # 이전과 다른 행동이 새로 감지된 경우만 기록
                if label != previous_label:
                    action = top_pred.get("action") or ACTION_MAP.get(label, "알 수 없는 행동")

                    timeline_events.append({
                        "timestamp": timestamp,
                        "label": label,
                        "action": action,
                        "confidence": top_pred.get("confidence", 0.0),
                        "bbox": top_pred.get("bbox"),
                        "sequence_count": top_pred.get("sequence_count", 0),
                        "sequence_required": top_pred.get("sequence_required", 60),
                        "image": frame_result["image"],
                        "predictions": [top_pred],
                    })

                    previous_label = label

            if timeline_events:
                for idx, event in enumerate(timeline_events, start=1):
                    timestamp = event["timestamp"]
                    label = event["label"]
                    action = event["action"]
                    confidence = event["confidence"]

                    draw_image = draw_predictions(
                        event["image"],
                        event["predictions"]
                    )

                    with st.expander(f"{idx}. {timestamp:.1f}초 - {label} / {action} 감지"):
                        st.write(f"감지 시점: {timestamp:.1f}초")
                        st.write(f"행동 클래스: {label}")
                        st.write(f"행동 이름: {action}")
                        st.write(f"Confidence score: {confidence:.4f}")
                        st.write(f"Bounding box: {event['bbox']}")
                        st.image(draw_image, caption="bbox 시각화 프레임", use_container_width=True)

            else:
                st.info("타임라인에 표시할 행동 변화가 없습니다.")

with tab4:
    st.subheader("디버그 확인")

    if "analysis_result" not in st.session_state:
        st.warning("아직 분석 결과가 없습니다. 2번 탭에서 분석을 먼저 실행하세요.")
    else:
        result = st.session_state["analysis_result"]
        debug_frames = result.get("debug_frames", [])

        if not debug_frames:
            st.info("디버그 결과가 없습니다. 1번 탭에서 '디버그 결과 수집'을 켠 뒤 다시 분석하세요.")
        else:
            st.markdown(
                """
                <div class="info-card">
                이 탭은 임시 점검용입니다.<br>
                1단계: YOLO bbox + ByteTrack ID → 2단계: bbox crop → 3단계: MediaPipe skeleton → 4단계: 90프레임 buffer/행동 클래스 순서로 확인합니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

            frame_options = [f"Frame {d['frame_index']} / {d['timestamp']:.2f}초" for d in debug_frames]
            selected_idx = st.selectbox(
                "확인할 디버그 프레임",
                options=list(range(len(debug_frames))),
                format_func=lambda idx: frame_options[idx],
            )

            render_debug_frame(debug_frames[selected_idx])
