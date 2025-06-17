#face_recognize/face_utils.py
import face_recognition
import numpy as np
import base64


FACE_MATCH_THRESHOLD = 0.6

def decode_face_encoding(enc_b64: str):
    """Decode a base64 string into a numpy vector"""
    return np.frombuffer(base64.b64decode(enc_b64), dtype=np.float64)

def compare_face_to_database(known_encodings, target_encoding, tolerance=0.45):
    """compare target face codes with known employee pools"""
    matches = face_recognition.compare_faces(known_encodings, target_encoding, tolerance)
    if True in matches:
        index = matches.index(True)
        return index
    return None

def compare_face_to_database_all(known_encodings, target_encoding, tolerance=0.45):
    matches = face_recognition.compare_faces(known_encodings, target_encoding, tolerance)
    matched_indices = [i for i, matched in enumerate(matches) if matched]
    return matched_indices