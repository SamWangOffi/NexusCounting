from flask import Flask
from OutlookSync.outlook_sync_api import calendar_bp
from OutlookSync.outlook_auth_api import auth_bp 
from face_recognize.register_api import register_bp
from face_recognize.recognize_api import recognize_bp
from face_recognize.log_api import log_bp
from face_recognize.trigger_api import trigger_bp
from warning.warning_api import warning_bp  
from face_upload.face_upload_api import upload_bp

app = Flask(__name__)

# Register all blueprints 
app.register_blueprint(calendar_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(register_bp)
app.register_blueprint(recognize_bp)
app.register_blueprint(log_bp)
app.register_blueprint(trigger_bp)
app.register_blueprint(warning_bp)
app.register_blueprint(upload_bp)

if __name__ == "__main__":
    app.run(debug=True)
