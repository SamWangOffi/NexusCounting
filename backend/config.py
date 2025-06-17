# config.py

import os
from dotenv import load_dotenv

# Optional: load .env file if used
load_dotenv()

# === Database Configuration ===
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "Nexus_guide_DB"
DB_USER = "postgres"
DB_PASSWORD = "123456"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# === Outlook API Configuration ===
CLIENT_ID="x"
CLIENT_SECRET="x"
TENANT_ID="x"
# === RTSP Streams ===
RTSP_URL = "rtsp://10.100.124.105/0"
RTSP_URL_FACE = "rtsp://admin:admin@10.100.124.17/defaultPrimary?streamType=u"

# === Face Library Path ===
FACE_LIB_DIR = r"D:\Internship_Project\2025.06.06\Face_Lib"  # Development PC
# FACE_LIB_DIR = r"C:\Users\User\Desktop\Nexus_Smart_Guide\Face_Lib"  # Deployment PC
