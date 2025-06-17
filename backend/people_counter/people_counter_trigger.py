# dual_camera_optimized.py
# ✅ 最终版：FSM状态机 + 跳帧优化 + 单次人脸识别 + 离开检测，科学且稳定

import sys
import os
import cv2
import numpy as np
import base64
import psycopg
import face_recognition
import time
import requests
from datetime import datetime
from collections import deque, defaultdict
from ultralytics import YOLO
from people_counter.byte_tracker.byte_tracker import BYTETracker
from config import RTSP_URL_FACE, RTSP_URL, DB_NAME, DB_USER, DB_PASSWORD

# ==== 区域设置 ====
box_x1, box_y1 = 900, 500
box_x2, box_y2 = 1250, 880
region_box = (box_x1, box_y1, box_x2, box_y2)

# ==== 参数设置 ====
LEAVING_WAIT_TIME = 60
MOVEMENT_THRESHOLD = 20
HISTORY_FRAMES = 5
ALERT_THRESHOLD = 22
DETECTING_TIMEOUT = 10
COOLDOWN_SECONDS = 15
STATIONARY_TIME = 5
ENTERED_STABLE_SECONDS = 10
RECOGNITION_DURATION = 5
RECOGNITION_INTERVAL = 1
FSM_FRAME_SKIP = 3
CAM2_FRAME_SKIP = 2

TRIGGER_URL = "http://localhost:5000/api/face/trigger"
WARNING_URL = "http://localhost:5000/api/warning"
LOG_URL = "http://localhost:5000/api/log/face"

model = YOLO("yolov8n.pt").to("cuda")

class TrackerArgs:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False
    frame_rate = 15

tracker = BYTETracker(TrackerArgs())

# ==== 状态变量 ====
state = "WAITING"
last_state = None
has_recognized = False

entered_ids = set()
candidate_ids = set()
movement_history = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))
id_birth_place = dict()
path_started_in_box = dict()
current_detected_ids = set()
in_box_count = 0
id_last_seen_frame = {}
current_frame_idx = 0
id_stationary_start_time = {}
entering_stable_start_time = None
leaving_start_time = None
last_group_current_count = 0
last_warning_time = 0

def is_inside_red_box(x, y):
    return box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2

def update_movement(tid, cx, cy):
    movement_history[tid].append((cx, cy))

def is_stationary(tid):
    if len(movement_history[tid]) < HISTORY_FRAMES:
        return False
    points = np.array(movement_history[tid])
    dists = np.linalg.norm(points - points.mean(axis=0), axis=1)
    return np.max(dists) < MOVEMENT_THRESHOLD

def load_staff_encodings():
    conn = psycopg.connect(f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}")
    cur = conn.cursor()
    cur.execute("SELECT name, face_encoding FROM staff")
    data = cur.fetchall()
    cur.close()
    conn.close()
    names = []
    encodings = []
    for name, enc_b64 in data:
        try:
            enc = np.frombuffer(base64.b64decode(enc_b64), dtype=np.float64)
            names.append(name)
            encodings.append(enc)
        except:
            continue
    return names, encodings

def post_warning(count):
    global last_warning_time
    if time.time() - last_warning_time > COOLDOWN_SECONDS:
        try:
            print(f"[Warning] Count: {count} → POST /api/warning")
            requests.post(WARNING_URL, json={"count": count, "timestamp": datetime.now().isoformat()})
            last_warning_time = time.time()
        except Exception as e:
            print(f"[Warning] Error: {e}")

def log_recognized_faces(names):
    try:
        for name in names:
            print(f"🧠 Recognized: {name}")
            requests.post(LOG_URL, json={"name": name})
    except Exception as e:
        print(f"[Log] Error: {e}")

def recognize_and_check_leave():
    global leaving_start_time
    cap2 = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    known_names, known_encodings = load_staff_encodings()
    recognized_names_set = set()
    recognition_start_time = time.time()
    last_recog_time = 0
    cam2_frame_idx = 0

    while True:
        ret, frame2 = cap2.read()
        if not ret:
            print("❌ camera2 读取失败，重试中...")
            time.sleep(0.2)
            continue

        cam2_frame_idx += 1
        if cam2_frame_idx % CAM2_FRAME_SKIP != 0:
            continue

        now = time.time()
        if now - recognition_start_time <= RECOGNITION_DURATION:
            if now - last_recog_time >= RECOGNITION_INTERVAL:
                last_recog_time = now
                rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
                    if True in matches:
                        idx = matches.index(True)
                        recognized_names_set.add(known_names[idx])

        results2 = model(frame2, imgsz=640, classes=[0], verbose=False)[0]
        people_count_secondary = len(results2.boxes)
        left_threshold = 3 if len(entered_ids) >= 3 else 1

        if people_count_secondary < left_threshold:
            if leaving_start_time is None:
                leaving_start_time = now
            elif now - leaving_start_time > LEAVING_WAIT_TIME:
                cap2.release()
                log_recognized_faces(list(recognized_names_set))
                print("✅ GROUP LEFT")
                return True
        else:
            leaving_start_time = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap2.release()
    return False

def main():
    global state, last_state, entered_ids, candidate_ids
    global id_birth_place, path_started_in_box, in_box_count
    global id_last_seen_frame, current_frame_idx, id_stationary_start_time
    global entering_stable_start_time, last_group_current_count, has_recognized

    cap = cv2.VideoCapture(RTSP_URL_FACE, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        current_frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            print("❌ camera1 读取失败")
            break

        if current_frame_idx % FSM_FRAME_SKIP != 0:
            continue

        results = model(frame, imgsz=640, classes=[0], verbose=False)[0]
        dets = [[*box.xyxy[0].cpu().numpy(), float(box.conf[0])] for box in results.boxes]
        targets = tracker.update(np.array(dets, dtype=float), frame.shape, frame.shape) if dets else []

        current_detected_ids.clear()
        in_box_count = 0
        now = time.time()

        for t in targets:
            tid = int(t.track_id)
            x1, y1, w, h = t.tlwh
            cx, cy = x1 + w / 2, y1 + h / 2
            update_movement(tid, cx, cy)
            current_detected_ids.add(tid)

            if tid not in id_birth_place:
                if current_frame_idx - id_last_seen_frame.get(tid, -1000) > 10:
                    id_birth_place[tid] = (cx, cy)
                    path_started_in_box[tid] = is_inside_red_box(cx, cy)
                    if path_started_in_box[tid]:
                        candidate_ids.add(tid)
            id_last_seen_frame[tid] = current_frame_idx

            if tid in candidate_ids and not is_inside_red_box(cx, cy):
                entered_ids.add(tid)

            if tid in candidate_ids and is_inside_red_box(cx, cy):
                in_box_count += 1
                if is_stationary(tid):
                    if tid not in entered_ids:
                        if tid not in id_stationary_start_time:
                            id_stationary_start_time[tid] = now
                        elif now - id_stationary_start_time[tid] > STATIONARY_TIME:
                            entered_ids.add(tid)
                    else:
                        id_stationary_start_time.pop(tid, None)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
        cv2.putText(frame, f"State: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("RTSP Detection", frame_small)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if state != last_state:
            print(f"🟡 STATE CHANGE: {last_state} → {state} | Detected: {len(current_detected_ids)} | In-box: {in_box_count}")
            last_state = state

        if state == "WAITING":
            if current_detected_ids:
                state = "DETECTING"
                detecting_start_time = now

        elif state == "DETECTING":
            if in_box_count >= 1:
                state = "ENTERING"
                entering_stable_start_time = None
                last_group_current_count = 0
            elif now - detecting_start_time > DETECTING_TIMEOUT:
                state = "WAITING"
                entered_ids.clear()
                candidate_ids.clear()
                id_birth_place.clear()
                path_started_in_box.clear()

        elif state == "ENTERING":
            group_current_count = len(entered_ids)
            if group_current_count == last_group_current_count:
                if entering_stable_start_time is None:
                    entering_stable_start_time = now
                elif now - entering_stable_start_time > ENTERED_STABLE_SECONDS * FSM_FRAME_SKIP:
                    print(f"✅ GROUP ENTERED with {group_current_count} people")
                    state = "ENTERED"
                    if group_current_count >= ALERT_THRESHOLD:
                        post_warning(group_current_count)
            else:
                entering_stable_start_time = now
                last_group_current_count = group_current_count

        elif state == "ENTERED":
            if not has_recognized:
                print("🔍 Running face recognition for ENTERED group...")
                cap.release()
                left = recognize_and_check_leave()
                cap = cv2.VideoCapture(RTSP_URL_FACE, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                has_recognized = True
                if left:
                    state = "LEFT"

        elif state == "LEFT":
            state = "WAITING"
            has_recognized = False
            entered_ids.clear()
            candidate_ids.clear()
            id_birth_place.clear()
            path_started_in_box.clear()
            entering_stable_start_time = None
            leaving_start_time = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
