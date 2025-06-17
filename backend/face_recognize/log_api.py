# face_recognize/log_api.py
from flask import Blueprint, request, jsonify
from utils.db_utils import insert_to_db

log_bp = Blueprint("log", __name__)

@log_bp.route("/api/log/face", methods=["POST"])
def log_face():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"message": "Missing 'name'"}), 400

    sql = "INSERT INTO face_log (name) VALUES (%s)"
    success = insert_to_db(sql, (name,))
    if success:
        return jsonify({"message": "Log saved."})
    else:
        return jsonify({"message": "Failed to save log."}), 500
