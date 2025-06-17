# log_utils.py

import psycopg
from config import DATABASE_URL

def log_face_recognition(name):
    """Write identification records to the face_log table (auto timestamp)"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # ✅ 只插入 name，timestamp 自动生成
                cur.execute("INSERT INTO face_log (name) VALUES (%s)", (name,))
            conn.commit()
            print(f"[LOG] Logged recognition for {name}")
    except Exception as e:
        print(f"❌ Write to face_log failed: {e}")
