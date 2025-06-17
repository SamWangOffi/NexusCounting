#warning_api/warning_api.py
from flask import Blueprint, request, jsonify
from datetime import datetime
from utils.db_utils import insert_to_db, fetch_all  

warning_bp = Blueprint("warning", __name__)

@warning_bp.route("/api/warning", methods=["POST"])
def receive_warning():
    data = request.get_json()
    count = data.get("count")
    timestamp = data.get("timestamp", datetime.now().isoformat())

    if count is None:
        return jsonify({"error": "Missing 'count'"}), 400

    sql = "INSERT INTO warning_log (count, timestamp) VALUES (%s, %s)"
    success = insert_to_db(sql, (count, timestamp))

    if success:
        return jsonify({"status": "logged", "count": count, "timestamp": timestamp}), 200
    else:
        return jsonify({"error": "Failed to log warning"}), 500

@warning_bp.route("/api/warning", methods=["GET"])
def get_recent_warnings():
    try:
        sql = "SELECT count, timestamp FROM warning_log ORDER BY timestamp DESC LIMIT 10"
        rows = fetch_all(sql)
        return jsonify([{"count": row["count"], "timestamp": row["timestamp"].isoformat()} for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
