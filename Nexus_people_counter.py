import cv2
import os
import time
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker

# ==== RTSP Settings ====
rtsp_url = "rtsp://admin:admin@10.100.124.17/defaultPrimary?streamType=u"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
model = YOLO("yolov8n.pt")  # CPU mode

# ==== ByteTrack Settings ====
class TrackerArgs:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False
    frame_rate = 15

tracker = BYTETracker(TrackerArgs())

# ==== Red Box Settings ====
box_x1, box_y1 = 900, 500
box_x2, box_y2 = 1250, 880
region_box = (box_x1, box_y1, box_x2, box_y2)

# ==== Parameters ====
MIN_DURATION = 1.0
overlap_thresh_enter = 0.4
LEAVING_WAIT_TIME = 60
STABLE_TIME = 5
MOVEMENT_THRESHOLD = 20
HISTORY_FRAMES = 5
ALERT_THRESHOLD = 22
DETECTING_TIMEOUT = 30
REQUIRED_STABLE_FRAMES = 10

# FSM & Tracking Variables
state = "WAITING"
group_current_count = 0
group_total_count = 0
entered_ids = set()
entry_time = {}
leaving_start_time = None
stable_start_time = None
detecting_start_time = None
stable_frame_count = 0
movement_history = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))
id_birth_place = dict()
candidate_ids = set()
path_started_in_box = dict()
counting_locked = False
late_joiner_count = 0

last_online_targets = []
current_detected_ids = set()
in_box_count = 0

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
print("[System] Started. Waiting for detections...")


def is_inside_red_box(x, y):
    return box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2

def update_movement(tid, cx, cy):
    movement_history[tid].append((cx, cy))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame")
        time.sleep(0.3)
        continue

    results = model(frame, imgsz=640, classes=[0], verbose=False)[0]
    dets = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        dets.append([x1, y1, x2, y2, conf])

    last_online_targets = tracker.update(np.array(dets), frame.shape, frame.shape) if dets else []
    current_detected_ids.clear()
    in_box_count = 0
    now = time.time()

    for t in last_online_targets:
        tid = int(t.track_id)
        x1, y1, w, h = t.tlwh
        x2, y2 = x1 + w, y1 + h
        cx, cy = x1 + w / 2, y1 + h / 2

        update_movement(tid, cx, cy)
        current_detected_ids.add(tid)

        if tid not in id_birth_place:
            id_birth_place[tid] = (cx, cy)
            path_started_in_box[tid] = is_inside_red_box(cx, cy)
            if path_started_in_box[tid]:
                candidate_ids.add(tid)

        if tid in candidate_ids and is_inside_red_box(cx, cy):
            in_box_count += 1

        inter_x1 = max(x1, box_x1)
        inter_y1 = max(y1, box_y1)
        inter_x2 = min(x2, box_x2)
        inter_y2 = min(y2, box_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        bbox_area = (x2 - x1) * (y2 - y1)
        overlap_ratio = inter_area / bbox_area if bbox_area > 0 else 0

        if tid not in entered_ids and tid in candidate_ids and path_started_in_box.get(tid, False):
            if overlap_ratio > overlap_thresh_enter:
                if tid not in entry_time:
                    entry_time[tid] = now
                elif now - entry_time[tid] >= MIN_DURATION:
                    entered_ids.add(tid)
                    if not counting_locked:
                        group_current_count += 1
                        group_total_count = group_current_count
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ID {tid} Entered, Total: {group_total_count}")
                    else:
                        late_joiner_count += 1
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ID {tid} Late Entered, Late Total: {late_joiner_count}")
                        if group_total_count + late_joiner_count > ALERT_THRESHOLD:
                            print(f"⚠ ALERT: Combined total exceeded after late entry: {group_total_count + late_joiner_count}")

    if state == "WAITING":
        if current_detected_ids:
            state = "DETECTING"
            detecting_start_time = now
            print("[FSM] DETECTING...")

    elif state == "DETECTING":
        if in_box_count >= 1:
            state = "ENTERING"
            stable_start_time = now
            stable_frame_count = 0
            print("[FSM] ENTERING...")
        elif detecting_start_time and now - detecting_start_time > DETECTING_TIMEOUT:
            print("[FSM] DETECTING timeout, reset to WAITING...")
            state = "WAITING"
            group_current_count = 0
            group_total_count = 0
            entered_ids.clear()
            entry_time.clear()
            movement_history.clear()
            id_birth_place.clear()
            candidate_ids.clear()
            leaving_start_time = None
            stable_start_time = None
            detecting_start_time = None
            late_joiner_count = 0

    elif state == "ENTERING":
        if group_current_count == group_total_count:
            stable_frame_count += 1
            if (stable_frame_count >= REQUIRED_STABLE_FRAMES and
                now - stable_start_time >= STABLE_TIME and
                now - detecting_start_time >= STABLE_TIME and
                group_total_count > 0):
                state = "ENTERED"
                counting_locked = True
                print(f"[FSM] ENTERED. Final Group Total: {group_total_count}")
                if group_total_count >= ALERT_THRESHOLD:
                    print(f"⚠ ALERT: Group size exceeds {ALERT_THRESHOLD}: {group_total_count}")
        else:
            stable_start_time = now
            stable_frame_count = 0

    elif state == "ENTERED":
        if not current_detected_ids:
            if leaving_start_time is None:
                leaving_start_time = now
            elif now - leaving_start_time > LEAVING_WAIT_TIME:
                print("[FSM] LEFT. Resetting...")
                state = "LEFT"
        else:
            leaving_start_time = None

    elif state == "LEFT":
        state = "WAITING"
        group_current_count = 0
        group_total_count = 0
        entered_ids.clear()
        entry_time.clear()
        movement_history.clear()
        id_birth_place.clear()
        candidate_ids.clear()
        leaving_start_time = None
        stable_start_time = None
        detecting_start_time = None
        stable_frame_count = 0
        counting_locked = False
        late_joiner_count = 0
        print("[FSM] System reset to WAITING.")

    time.sleep(1)
