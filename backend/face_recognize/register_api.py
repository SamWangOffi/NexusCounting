from flask import Blueprint, request, jsonify
import os
import face_recognition
import numpy as np
import base64
from config import DATABASE_URL, FACE_LIB_DIR
from utils.db_utils import insert_to_db, fetch_one  

register_bp = Blueprint("register", __name__)

@register_bp.route("/api/staff/register", methods=["POST"])
@register_bp.route("/api/face/register", methods=["POST"])
def register_employee():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"message": "Missing 'name'"}), 400

    folder_name = name.replace(" ", "_")
    person_dir = os.path.join(FACE_LIB_DIR, folder_name)
    if not os.path.isdir(person_dir):
        return jsonify({"message": f"No such folder: {person_dir}"}), 404

    encodings = []
    for file in os.listdir(person_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(person_dir, file)
            img = face_recognition.load_image_file(img_path)
            faces = face_recognition.face_encodings(img)
            if faces:
                encodings.append(faces[0])

    if not encodings:
        return jsonify({"message": f"No valid face encodings found for {name}."}), 400

    avg_encoding = np.mean(encodings, axis=0)
    encoding_b64 = base64.b64encode(avg_encoding.tobytes()).decode('utf-8')

    
    check_sql = "SELECT 1 FROM staff WHERE name = %s"
    existing = fetch_one(check_sql, (name,))
    if existing:
        return jsonify({"message": f"{name} already registered."})

   
    insert_sql = "INSERT INTO staff (name, face_encoding) VALUES (%s, %s)"
    success = insert_to_db(insert_sql, (name, encoding_b64))
    if success:
        return jsonify({"message": f"{name} registered successfully."})
    else:
        return jsonify({"message": f"Failed to register {name}."}), 500