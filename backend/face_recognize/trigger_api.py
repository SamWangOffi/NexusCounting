# trigger_api.py
from flask import Blueprint, jsonify
import requests

trigger_bp = Blueprint("trigger", __name__)

@trigger_bp.route("/api/face/trigger", methods=["POST"])
def trigger_face_recognition():
    try:
        print("[Trigger] Sending GET to /api/face/recognize")
        res = requests.get("http://localhost:5000/api/face/recognize")
        return jsonify({"status": "triggered", "face_response": res.json()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
