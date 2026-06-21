from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from ultralytics import YOLO
from collections import defaultdict, deque
import io
import base64
import json
import os
import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except ImportError as e:
    mp = None
    MEDIAPIPE_IMPORT_ERROR = e
else:
    MEDIAPIPE_IMPORT_ERROR = None

app = FastAPI(title="Salpim Action Recognition API - ONNX Fusion")

# =========================
# 기본 설정
# =========================

SEQUENCE_LENGTH = 90
RGB_FRAME_COUNT = 16
NUM_JOINTS = 17
IN_CHANNELS = 6
IMAGE_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ONNX_MODEL_PATH = os.getenv(
    "ONNX_MODEL_PATH",
    os.path.join(MODEL_DIR, "full_fusion_single.onnx"),
)
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", os.path.join(MODEL_DIR, "label_map.json"))

DEFAULT_LABELS = ["A010", "A011", "A016", "A018", "A023", "A031", "A035", "A041", "A053", "A054"]

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
}
RISK_LABELS = {"A053"}

# 최종 전처리에서 사용한 MediaPipe 17관절 순서
SELECTED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 5, 2, 7, 8]
LEFT_HIP_IDX = 7
RIGHT_HIP_IDX = 8
LEFT_SHOULDER_IDX = 1
RIGHT_SHOULDER_IDX = 2
EPS = 1e-6

# ONNX input/output name
IMAGE_INPUT_NAME = "images"       # [batch_size, 16, 3, 224, 224]
SKELETON_INPUT_NAME = "skeleton"  # [batch_size, 6, 90, 17]
OUTPUT_NAME = "output"            # [batch_size, 10]

idx_to_class = None
onnx_session = None
detector = None
pose = None

# session별 버퍼
# RGB는 전체 프레임 기준으로 저장
session_frame_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
# Skeleton은 track_id별로 저장
session_skeleton_sequences = defaultdict(lambda: defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH)))
session_prev_positions = defaultdict(dict)

PREDICTION_WINDOW = 5
MIN_SAME_LABEL = 1
RISK_CONFIRM_COUNT = 1
session_label_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=PREDICTION_WINDOW)))
session_risk_streak = defaultdict(lambda: defaultdict(int))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# =========================
# 모델 로딩
# =========================

def load_label_map():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        first_key = next(iter(label_map.keys()))
        if str(first_key).isdigit():
            return {int(k): v for k, v in label_map.items()}
        return {int(v): k for k, v in label_map.items()}
    return {i: label for i, label in enumerate(DEFAULT_LABELS)}


def get_onnx_providers():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_all_models():
    global idx_to_class, onnx_session, detector, pose

    if mp is None:
        raise ImportError(f"mediapipe import failed: {MEDIAPIPE_IMPORT_ERROR}")
    if not os.path.exists(ONNX_MODEL_PATH):
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")

    idx_to_class = load_label_map()

    providers = get_onnx_providers()
    onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

    input_info = {inp.name: inp.shape for inp in onnx_session.get_inputs()}
    print("ONNX providers:", onnx_session.get_providers())
    print("ONNX inputs:", input_info)
    print("ONNX outputs:", {out.name: out.shape for out in onnx_session.get_outputs()})

    detector = YOLO("yolov8n.pt")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


@app.on_event("startup")
def startup_event():
    try:
        print("모델 로딩 시작")
        load_all_models()
        print("모델 로딩 완료")
    except Exception as e:
        import traceback
        print("모델 로딩 실패")
        print(repr(e))
        traceback.print_exc()
        raise


# =========================
# 전처리 함수
# =========================

def extract_mediapipe_positions(frame_bgr):
    """단일 인물 영상 기준으로 전체 프레임에서 MediaPipe skeleton을 추출합니다."""
    if pose is None:
        raise RuntimeError("MediaPipe pose model is not loaded.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        return None, None

    h, w = frame_bgr.shape[:2]
    coords = []
    pixel_coords = []
    for lm_idx in SELECTED_LANDMARKS:
        lm = results.pose_landmarks.landmark[lm_idx]
        coords.append([lm.x, lm.y, lm.z])
        pixel_coords.append([lm.x * w, lm.y * h])

    coords = np.array(coords, dtype=np.float32)
    pixel_coords = np.array(pixel_coords, dtype=np.float32)
    return coords, pixel_coords


def normalize_pose(coords):
    hip_center = (coords[LEFT_HIP_IDX] + coords[RIGHT_HIP_IDX]) / 2.0
    shoulder_center = (coords[LEFT_SHOULDER_IDX] + coords[RIGHT_SHOULDER_IDX]) / 2.0
    scale = np.linalg.norm(shoulder_center - hip_center)
    if scale < EPS:
        scale = 1.0
    return (coords - hip_center) / scale


def make_6channel_features(session_id, track_id, coords):
    pos = normalize_pose(coords)
    prev_positions = session_prev_positions[session_id]

    if track_id not in prev_positions:
        velocity = np.zeros_like(pos)
    else:
        velocity = pos - prev_positions[track_id]

    prev_positions[track_id] = pos.copy()
    return np.concatenate([pos, velocity], axis=1).astype(np.float32)  # [17, 6]


def preprocess_single_frame_bgr(frame_bgr):
    """전체 프레임 1장을 ONNX RGB 입력용 [3,224,224]로 변환합니다."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    arr = frame_rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.astype(np.float32)


def make_images_tensor(frame_buffer):
    """최근 90프레임에서 16장을 균등 샘플링해 [1,16,3,224,224] 생성."""
    frames = list(frame_buffer)
    if len(frames) < RGB_FRAME_COUNT:
        raise ValueError(f"RGB frame buffer is too short: {len(frames)}")

    # 90프레임이 모이면 최근 90프레임 전체에서 16장 샘플링
    # 90프레임 전에는 현재까지 누적된 프레임에서 16장 샘플링 가능하지만
    # 실제 추론은 skeleton 90프레임이 모인 뒤에만 수행됨
    idxs = np.linspace(0, len(frames) - 1, RGB_FRAME_COUNT).astype(int)
    sampled = [preprocess_single_frame_bgr(frames[i]) for i in idxs]
    images = np.stack(sampled, axis=0)  # [16,3,224,224]
    return np.expand_dims(images, axis=0).astype(np.float32)


def make_skeleton_tensor(sequence):
    seq = np.array(sequence, dtype=np.float32)  # [90,17,6]
    if seq.shape != (SEQUENCE_LENGTH, NUM_JOINTS, IN_CHANNELS):
        raise ValueError(f"skeleton sequence shape error: {seq.shape}")
    seq = np.transpose(seq, (2, 0, 1))  # [6,90,17]
    return np.expand_dims(seq, axis=0).astype(np.float32)


def softmax_np(logits):
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def run_onnx_inference(images_tensor, skeleton_tensor):
    outputs = onnx_session.run(
        [OUTPUT_NAME],
        {
            IMAGE_INPUT_NAME: images_tensor,
            SKELETON_INPUT_NAME: skeleton_tensor,
        },
    )[0]
    prob = softmax_np(outputs)
    pred_idx = int(np.argmax(prob, axis=1)[0])
    confidence = float(prob[0, pred_idx])
    return pred_idx, confidence


# =========================
# 시각화 및 디버그 함수
# =========================

def encode_debug_image_b64(image_bgr, max_width=480):
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        image_bgr = cv2.resize(image_bgr, (max_width, max(1, int(h * scale))))

    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def draw_skeleton_debug(frame_bgr, pixel_coords):
    debug_img = frame_bgr.copy()
    if pixel_coords is None:
        return debug_img

    for px, py in pixel_coords:
        cv2.circle(debug_img, (int(px), int(py)), 5, (0, 255, 0), -1)
    return debug_img


def build_debug_payload(
    *,
    enabled,
    stage,
    track_id,
    bbox,
    original_bbox,
    frame,
    pixel_coords=None,
    pose_detected=False,
    message="",
):
    if not enabled:
        return None

    frame_h, frame_w = frame.shape[:2] if frame is not None and frame.size > 0 else (0, 0)
    keypoints = []
    if pixel_coords is not None:
        keypoints = [[float(px), float(py)] for px, py in pixel_coords]

    skeleton_img = None
    if frame is not None and frame.size > 0:
        skeleton_img = draw_skeleton_debug(frame, pixel_coords)

    return {
        "stage": stage,
        "message": message,
        "track_id": int(track_id) if track_id is not None else None,
        "bbox": [int(v) for v in bbox] if bbox is not None else None,
        "original_bbox": [int(v) for v in original_bbox] if original_bbox is not None else None,
        "crop_size": {"width": int(frame_w), "height": int(frame_h)},
        "pose_detected": bool(pose_detected),
        "crop_keypoints": keypoints,
        # app.py 디버그 UI 호환을 위해 기존 key 이름 유지
        "crop_image_b64": encode_debug_image_b64(frame),
        "crop_skeleton_image_b64": encode_debug_image_b64(skeleton_img),
    }


def select_primary_track(boxes, track_ids, image_width, image_height):
    """단일 인물 영상 기준으로 가장 큰 bbox를 대표 track으로 선택합니다."""
    if len(track_ids) == 0:
        return None, None

    best_idx = None
    best_area = -1

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_width, x2), min(image_height, y2)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best_idx = i

    if best_idx is None:
        return None, None
    return boxes[best_idx], int(track_ids[best_idx])


# =========================
# API
# =========================

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "sequence_length": SEQUENCE_LENGTH,
        "rgb_frame_count": RGB_FRAME_COUNT,
        "num_joints": NUM_JOINTS,
        "model_loaded": onnx_session is not None,
        "pose_backend": "mediapipe_full_frame",
        "onnx_model": ONNX_MODEL_PATH,
        "onnx_providers": onnx_session.get_providers() if onnx_session is not None else [],
        "onnx_inputs": [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in onnx_session.get_inputs()
        ] if onnx_session is not None else [],
    }


@app.post("/reset")
def reset_session(session_id: str = Form("default")):
    for store in [
        session_frame_buffers,
        session_skeleton_sequences,
        session_prev_positions,
        session_label_history,
        session_risk_streak,
    ]:
        if session_id in store:
            del store[session_id]
    return {"status": "reset", "session_id": session_id}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    debug: bool = Form(False),
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]

        # ONNX images 입력은 bbox crop이 아니라 전체 프레임 기준
        frame_buffer = session_frame_buffers[session_id]
        frame_buffer.append(frame.copy())

        results = detector.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
        predictions = []
        debug_entries = []

        if len(results) == 0 or results[0].boxes is None or results[0].boxes.id is None:
            return {
                "filename": file.filename,
                "image_size": {"width": width, "height": height},
                "predictions": [],
                "debug": {"enabled": debug, "entries": []},
            }

        boxes = results[0].boxes
        track_ids = boxes.id.int().cpu().tolist()
        xyxy_list = boxes.xyxy.cpu().numpy()

        selected_box, track_id = select_primary_track(xyxy_list, track_ids, width, height)
        if selected_box is None:
            return {
                "filename": file.filename,
                "image_size": {"width": width, "height": height},
                "predictions": [],
                "debug": {"enabled": debug, "entries": []},
            }

        ox1, oy1, ox2, oy2 = map(int, selected_box)
        x1, y1 = max(0, ox1), max(0, oy1)
        x2, y2 = min(width, ox2), min(height, oy2)
        original_bbox = [ox1, oy1, ox2, oy2]
        clipped_bbox = [x1, y1, x2, y2]

        if x2 <= x1 or y2 <= y1:
            debug_entries.append(build_debug_payload(
                enabled=debug,
                stage="bbox_invalid",
                track_id=track_id,
                bbox=clipped_bbox,
                original_bbox=original_bbox,
                frame=frame,
                message="YOLO bbox가 이미지 경계 밖이거나 크기가 0입니다.",
            ))
            return {
                "filename": file.filename,
                "image_size": {"width": width, "height": height},
                "predictions": [],
                "debug": {"enabled": debug, "entries": [e for e in debug_entries if e is not None]},
            }

        # 단일 인물 영상 기준: MediaPipe는 전체 프레임에서 1회만 추출
        coords, pixel_coords = extract_mediapipe_positions(frame)
        if coords is None:
            debug_entries.append(build_debug_payload(
                enabled=debug,
                stage="pose_failed",
                track_id=track_id,
                bbox=clipped_bbox,
                original_bbox=original_bbox,
                frame=frame,
                pose_detected=False,
                message="전체 프레임에서 MediaPipe skeleton을 찾지 못했습니다.",
            ))
            return {
                "filename": file.filename,
                "image_size": {"width": width, "height": height},
                "predictions": [],
                "debug": {"enabled": debug, "entries": [e for e in debug_entries if e is not None]},
            }

        features = make_6channel_features(session_id, track_id, coords)
        if features.shape != (NUM_JOINTS, IN_CHANNELS):
            raise ValueError(f"feature shape error: {features.shape}")

        skeleton_sequence = session_skeleton_sequences[session_id][track_id]
        skeleton_sequence.append(features)

        label = raw_label = smoothed_label = "collecting"
        action = raw_action = smoothed_action = "프레임 수집 중"
        confidence = raw_confidence = 0.0
        ready = len(skeleton_sequence) >= SEQUENCE_LENGTH and len(frame_buffer) >= SEQUENCE_LENGTH
        is_risk = False

        if ready:
            images_tensor = make_images_tensor(frame_buffer)
            skeleton_tensor = make_skeleton_tensor(skeleton_sequence)

            pred_idx, raw_confidence = run_onnx_inference(images_tensor, skeleton_tensor)
            raw_label = idx_to_class[pred_idx]
            raw_action = ACTION_MAP.get(raw_label, "알 수 없는 행동")

            label_history = session_label_history[session_id][track_id]
            label_history.append(raw_label)
            recent_labels = list(label_history)
            last_label = recent_labels[-1]
            smoothed_label = last_label if recent_labels.count(last_label) >= MIN_SAME_LABEL else raw_label
            smoothed_action = ACTION_MAP.get(smoothed_label, "알 수 없는 행동")

            if smoothed_label in RISK_LABELS:
                session_risk_streak[session_id][track_id] += 1
            else:
                session_risk_streak[session_id][track_id] = 0

            is_risk = session_risk_streak[session_id][track_id] >= RISK_CONFIRM_COUNT
            label, action, confidence = smoothed_label, smoothed_action, raw_confidence

        keypoints = []
        for px, py in pixel_coords:
            keypoints.append([float(px), float(py)])

        debug_entry = build_debug_payload(
            enabled=debug,
            stage="ok",
            track_id=track_id,
            bbox=clipped_bbox,
            original_bbox=original_bbox,
            frame=frame,
            pixel_coords=pixel_coords,
            pose_detected=True,
            message="전체 프레임 RGB buffer와 전체 프레임 MediaPipe skeleton 추출 성공",
        )
        if debug_entry is not None:
            debug_entry.update({
                "sequence_count": len(skeleton_sequence),
                "sequence_required": SEQUENCE_LENGTH,
                "rgb_sequence_count": len(frame_buffer),
                "rgb_sequence_required": SEQUENCE_LENGTH,
                "ready": ready,
                "label": label,
                "action": action,
                "confidence": confidence,
                "raw_label": raw_label,
                "raw_action": raw_action,
                "raw_confidence": raw_confidence,
                "is_risk": is_risk,
            })
        debug_entries.append(debug_entry)

        predictions.append({
            "track_id": int(track_id),
            "label": label,
            "action": action,
            "confidence": confidence,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "keypoints": keypoints,
            "sequence_count": len(skeleton_sequence),
            "sequence_required": SEQUENCE_LENGTH,
            "rgb_sequence_count": len(frame_buffer),
            "rgb_sequence_required": SEQUENCE_LENGTH,
            "ready": ready,
            "raw_label": raw_label,
            "raw_action": raw_action,
            "raw_confidence": raw_confidence,
            "smoothed_label": smoothed_label,
            "smoothed_action": smoothed_action,
            "is_risk": is_risk,
        })

        debug_entries = [entry for entry in debug_entries if entry is not None]
        return {
            "filename": file.filename,
            "image_size": {"width": width, "height": height},
            "predictions": predictions,
            "debug": {"enabled": debug, "entries": debug_entries},
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "predictions": []}
