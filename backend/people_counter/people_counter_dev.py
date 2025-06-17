# people_counter_dev.py
from flask import Flask, jsonify, request
from threading import Thread
import time
import cv2
import numpy as np
from ultralytics import YOLO
from people_counter.byte_tracker.byte_tracker import BYTETracker
from collections import deque, defaultdict
from datetime import datetime
import requests
import os

from config import RTSP_URL_FACE

app = Flask(__name__)

# ==== Red Box Settings ====
box_x1, box_y1 = 900, 500
box_x2, box_y2 = 1250, 880

# ==== Parameters ====
LEAVING_WAIT_TIME = 60
STABLE_TIME = 3
MOVEMENT_THRESHOLD = 20
HISTORY_FRAMES = 5
ALERT_THRESHOLD = 22
DETECTING_TIMEOUT = 30
REQUIRED_STABLE_FRAMES = 3
COOLDOWN_SECONDS = 15
STATIONARY_TIME = 5

TRIGGER_URL = "http://localhost:5000/api/face/trigger"
WARNING_URL = "http://localhost:5000/api/warning"
CAMERA2_COUNT_API = "http://localhost:5001/api/camera2/count"

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

# ==== FSM 状态变量 ====
state = "WAITING"
last_state = None
fsm_running = False

# ==== 缓存变量 ====
group_current_count = 0
group_total_count = 0
entered_ids = set()
candidate_ids = set()
movement_history = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))
id_birth_place = dict()
path_started_in_box = dict()
current_detected_ids = set()
in_box_count = 0
stable_frame_count = 0
last_trigger_time = 0
last_warning_time = 0
leaving_start_time = None
stable_start_time = None
detecting_start_time = None
id_last_seen_frame = {}
current_frame_idx = 0
id_stationary_start_time = {}

def is_inside_red_box(x, y):
    return box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2

def update_movement(tid, cx, cy):
    movement_history[tid].append((cx, cy))

def post_trigger():
    global last_trigger_time
    if time.time() - last_trigger_time > COOLDOWN_SECONDS:
        try:
            requests.post(TRIGGER_URL)
            last_trigger_time = time.time()
        except: pass

def post_warning(count):
    global last_warning_time
    if time.time() - last_warning_time > COOLDOWN_SECONDS:
        try:
            requests.post(WARNING_URL, json={"count": count, "timestamp": datetime.now().isoformat()})
            last_warning_time = time.time()
        except: pass

def is_stationary(tid):
    if len(movement_history[tid]) < HISTORY_FRAMES:
        return False
    points = np.array(movement_history[tid])
    dists = np.linalg.norm(points - points.mean(axis=0), axis=1)
    return np.max(dists) < MOVEMENT_THRESHOLD

def get_camera2_people_count():
    try:
        resp = requests.get(CAMERA2_COUNT_API, timeout=1)
        if resp.ok:
            return resp.json().get("count", 0)
    except:
        return 0

def run_fsm():
    global state, last_state, group_current_count, group_total_count, entered_ids
    global leaving_start_time, stable_start_time, detecting_start_time
    global stable_frame_count, id_birth_place, candidate_ids, path_started_in_box
    global current_detected_ids, in_box_count, id_last_seen_frame, current_frame_idx
    global id_stationary_start_time, fsm_running

    cap = cv2.VideoCapture(RTSP_URL_FACE, cv2.CAP_FFMPEG)
    while fsm_running:
        current_frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, classes=[0], verbose=False)[0]
        dets = [[*box.xyxy[0].cpu().numpy(), float(box.conf[0])] for box in results.boxes]

        last_online_targets = tracker.update(np.array(dets, dtype=float), frame.shape, frame.shape) if dets else []
        current_detected_ids.clear()
        in_box_count = 0
        now = time.time()

        for t in last_online_targets:
            tid = int(t.track_id)
            x1, y1, w, h = t.tlwh
            cx, cy = x1 + w / 2, y1 + h / 2
            update_movement(tid, cx, cy)
            current_detected_ids.add(tid)

            if tid not in id_birth_place:
                last_seen = id_last_seen_frame.get(tid, -1000)
                if current_frame_idx - last_seen > 10:
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

        if state == "WAITING":
            if current_detected_ids:
                state = "DETECTING"
                detecting_start_time = now

        elif state == "DETECTING":
            if in_box_count >= 1:
                state = "ENTERING"
                stable_start_time = now
                stable_frame_count = 0
            elif detecting_start_time and now - detecting_start_time > DETECTING_TIMEOUT:
                state = "WAITING"
                entered_ids.clear()
                candidate_ids.clear()
                id_birth_place.clear()
                path_started_in_box.clear()

        elif state == "ENTERING":
            group_total_count = len(candidate_ids)
            group_current_count = len(entered_ids)

            if group_total_count > 0 and group_current_count / group_total_count >= 0.7:
                if stable_frame_count == 0:
                    stable_start_time = now
                stable_frame_count += 1
                if stable_frame_count >= REQUIRED_STABLE_FRAMES and now - stable_start_time >= STABLE_TIME:
                    state = "ENTERED"
                    post_trigger()
                    if group_total_count >= ALERT_THRESHOLD:
                        post_warning(group_total_count)
            else:
                stable_frame_count = 0
                stable_start_time = now

        elif state == "ENTERED":
            people_count_secondary = get_camera2_people_count()
            left_threshold = 3 if group_total_count >= 3 else 1
            if people_count_secondary < left_threshold:
                if leaving_start_time is None:
                    leaving_start_time = now
                elif now - leaving_start_time > LEAVING_WAIT_TIME:
                    state = "LEFT"
            else:
                leaving_start_time = None

        elif state == "LEFT":
            state = "WAITING"
            entered_ids.clear()
            candidate_ids.clear()
            id_birth_place.clear()
            path_started_in_box.clear()
            stable_frame_count = 0
            leaving_start_time = None

    cap.release()

@app.route("/api/people_counter/start", methods=["GET"])
def start_counter():
    global fsm_running
    if not fsm_running:
        fsm_running = True
        thread = Thread(target=run_fsm)
        thread.start()
        return jsonify({"message": "People counter started"})
    return jsonify({"message": "Already running"})

@app.route("/api/people_counter/status", methods=["GET"])
def get_status():
    return jsonify({
        "state": state,
        "entered": len(entered_ids),
        "total_candidates": len(candidate_ids)
    })

@app.route("/api/people/update_status", methods=["POST"])
def update_fsm_status():
    global state, leaving_start_time
    data = request.get_json()
    if data.get("status") == "LEFT" and state == "ENTERED":
        print("🔁 FSM external trigger received: FORCING → LEFT")
        state = "LEFT"
        leaving_start_time = None  # 清除原定计时器
        return jsonify({"message": "FSM moved to LEFT by external trigger"})
    return jsonify({"message": "No action taken"})


if __name__ == "__main__":
    app.run(debug=True)