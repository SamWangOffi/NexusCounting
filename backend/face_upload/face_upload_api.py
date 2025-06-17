#face_upload/face_upload_api.py

from flask import Blueprint, request, jsonify
import os
import face_recognition
import numpy as np
import psycopg
import uuid
import shutil
import base64
from datetime import datetime
from config import DATABASE_URL, FACE_LIB_DIR
from face_recognize.face_utils import FACE_MATCH_THRESHOLD

upload_bp = Blueprint("upload", __name__)

# Save image to staff folder with automatic numbering
def save_image_from_path(temp_path, name, original_ext):
    folder_path = os.path.join(FACE_LIB_DIR, name)
    os.makedirs(folder_path, exist_ok=True)
    existing_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    next_index = len(existing_files) + 1
    new_filename = f"{next_index}{original_ext}"
    final_path = os.path.join(folder_path, new_filename)
    shutil.copy(temp_path, final_path)
    return final_path, new_filename

# Extract face encoding from image file path
def extract_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

# Load all known staff encodings from database
def load_all_staff_encodings():
    names = []
    encodings = []
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name, face_encoding FROM staff")
            for name, b64_encoding in cur.fetchall():
                try:
                    encoding = np.frombuffer(base64.b64decode(b64_encoding), dtype=np.float64)
                    if encoding.shape != (128,):
                        print(f"Skipping invalid encoding for: {name} ({encoding.shape})")
                        continue
                    names.append(name)
                    encodings.append(encoding)
                except Exception as e:
                    print(f"Failed decoding face_encoding for {name}: {e}")
    return names, encodings

# Insert or update staff encoding in database
def update_staff_encoding(name, new_encoding):
    if new_encoding.shape != (128,):
        print(f"Refusing to update {name} with invalid new encoding shape: {new_encoding.shape}")
        return

    encoding_b64 = base64.b64encode(new_encoding.tobytes()).decode("utf-8")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT face_encoding FROM staff WHERE name = %s", (name,))
            result = cur.fetchone()
            if result:
                try:
                    old_encoding = np.frombuffer(base64.b64decode(result[0]), dtype=np.float64)
                    if old_encoding.shape != (128,):
                        print(f"Detected invalid shape for {name}: {old_encoding.shape}. Overwriting with new encoding.")
                        cur.execute("UPDATE staff SET face_encoding = %s WHERE name = %s", (encoding_b64, name))
                    else:
                        avg_encoding = np.mean([old_encoding, new_encoding], axis=0)
                        encoding_b64 = base64.b64encode(avg_encoding.tobytes()).decode("utf-8")
                        cur.execute("UPDATE staff SET face_encoding = %s WHERE name = %s", (encoding_b64, name))
                except Exception as e:
                    print(f"Failed processing existing encoding for {name}: {e}")
            else:
                cur.execute("INSERT INTO staff (name, face_encoding) VALUES (%s, %s)", (name, encoding_b64))
            conn.commit()

# Main upload API
@upload_bp.route("/api/face/upload", methods=["POST"])
def upload_face():
    print("Received upload request")

    file = request.files.get("photo")
    name = request.form.get("name")
    print(f"Filename: {file.filename if file else 'None'}, Name: {name}")

    if not file:
        return jsonify({"error": "Missing photo file."}), 400

    original_ext = os.path.splitext(file.filename)[-1].lower()
    temp_filename = str(uuid.uuid4()) + original_ext
    os.makedirs("temp_upload", exist_ok=True)
    temp_path = os.path.join("temp_upload", temp_filename)
    file.save(temp_path)

    encoding = extract_encoding(temp_path)
    if encoding is None or encoding.shape != (128,):
        os.remove(temp_path)
        print(f"Invalid face encoding. Shape: {None if encoding is None else encoding.shape}")
        return jsonify({"error": "Invalid face encoding."}), 400

    known_names, known_encodings = load_all_staff_encodings()
    if known_encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        min_idx = np.argmin(distances)
        if distances[min_idx] < FACE_MATCH_THRESHOLD:
            matched_name = known_names[min_idx]
            file_path, saved_filename = save_image_from_path(temp_path, matched_name, original_ext)
            update_staff_encoding(matched_name, encoding)
            os.remove(temp_path)
            print(f"Recognized as existing staff: {matched_name}, saved {saved_filename}")
            return jsonify({"result": "recognized", "name": matched_name})

    if not name:
        os.remove(temp_path)
        return jsonify({"error": "Unrecognized face. Please provide a name to register."}), 400

    name = name.strip().replace(" ", "_")
    file_path, saved_filename = save_image_from_path(temp_path, name, original_ext)
    update_staff_encoding(name, encoding)
    os.remove(temp_path)
    print(f"New staff registered: {name}, saved {saved_filename}")
    return jsonify({"result": "new", "name": name})
