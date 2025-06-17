# face_api/recognize_api.py
from flask import Blueprint, jsonify
import cv2
import face_recognition
import numpy as np
import psycopg
import time
import requests
from datetime import datetime
from config import DATABASE_URL, RTSP_URL
from face_recognize.face_utils import decode_face_encoding
from utils.log_utils import log_face_recognition
from ultralytics import YOLO

recognize_bp = Blueprint("recognize", __name__)

# 加载员工编码
def load_staff_faces():
    names = []
    encodings = []
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name, face_encoding FROM staff")
            for name, enc_b64 in cur.fetchall():
                try:
                    enc = decode_face_encoding(enc_b64)
                    if not isinstance(enc, np.ndarray) or enc.shape != (128,):
                        print(f"⚠️ Skipping invalid encoding for {name}: {enc.shape if isinstance(enc, np.ndarray) else type(enc)}")
                        continue
                    names.append(name)
                    encodings.append(enc)
                except Exception as e:
                    print(f"❌ Error decoding encoding for {name}: {e}")
    print("🧠 Loaded staff:", names)
    return names, encodings

# 查询当前时间内的事件
def query_current_events_for(name):
    now = datetime.now()
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, start_time, end_time, guide_name, description
                FROM calendar_event
                WHERE guide_name = %s
                AND start_time <= (%s + interval '15 minutes')
                AND end_time >= (%s - interval '15 minutes')
            """, (name, now, now))
            return cur.fetchall()

# YOLO模型加载一次
yolo_model = YOLO("yolov8n.pt").to("cuda")

# 全局变量记录低人数开始时间
low_people_start_time = None
LEFT_TRIGGER_SECONDS = 3

@recognize_bp.route("/api/face/recognize", methods=["GET"])
def recognize_faces():
    global low_people_start_time
    FACE_MATCH_THRESHOLD = 0.55

    names, encodings = load_staff_faces()
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        return jsonify({"error": "Failed to open camera"}), 500

    time.sleep(2)

    detected = []
    already_logged = set()
    recognized_with_events = []

    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            continue
        if i % 3 != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        faces = face_recognition.face_encodings(rgb, locs)

        for enc in faces:
            if not encodings:
                continue
            distances = face_recognition.face_distance(encodings, enc)
            best_idx = np.argmin(distances)
            best_distance = distances[best_idx]

            if best_distance < FACE_MATCH_THRESHOLD:
                name = names[best_idx].replace("_", " ")
                if name not in already_logged:
                    already_logged.add(name)
                    detected.append(name)
                    log_face_recognition(name)

                    try:
                        events = query_current_events_for(name)
                        for title, start, end, guide, desc in events:
                            print(f"👋 Welcome {name} from \"{title}\" ({start.strftime('%H:%M')} ~ {end.strftime('%H:%M')})")
                            recognized_with_events.append({
                                "name": name,
                                "title": title,
                                "start_time": start.isoformat(),
                                "end_time": end.isoformat(),
                                "description": desc
                            })
                    except Exception as e:
                        print(f"Failed to query event for {name}: {e}")
            else:
                print(f"❌ No match. Closest is {names[best_idx]} (distance {best_distance:.4f})")

        # ========== YOLO 检测人数 ==========
        results = yolo_model(frame)[0]
        person_count = sum(1 for cls in results.boxes.cls if int(cls) == 0)
        print(f"👥 YOLO counted {person_count} people")

        # 检查是否触发 LEFT
        if person_count <= 2:
            if low_people_start_time is None:
                low_people_start_time = time.time()
            elif time.time() - low_people_start_time >= LEFT_TRIGGER_SECONDS:
                try:
                    print("📩 Sending FSM LEFT trigger")
                    requests.post("http://localhost:5000/api/people/update_status", json={"status": "LEFT"})
                except Exception as e:
                    print(f"❌ Failed to send LEFT trigger: {e}")
                low_people_start_time = None
        else:
            low_people_start_time = None

    cap.release()
    print("🔍 Recognized:", detected)
    return jsonify({
        "recognized": list(set(detected)),
        "events": recognized_with_events
    })
