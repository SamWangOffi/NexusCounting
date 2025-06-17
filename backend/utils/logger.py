# utils/db_utils.py

import psycopg
from config import DATABASE_URL
from psycopg.rows import dict_row

def get_connection():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def insert_to_db(sql, values):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
                conn.commit()
                return True
    except Exception as e:
        print(f"[DB] Insert error: {e}")
        return False

def fetch_one(sql, values=None):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
                return cur.fetchone()
    except Exception as e:
        print(f"[DB] Fetch one error: {e}")
        return None

def fetch_all(sql, values=None):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
                return cur.fetchall()
    except Exception as e:
        print(f"[DB] Fetch all error: {e}")
        return []
