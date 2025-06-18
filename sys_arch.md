# 系统架构图

```mermaid
graph TD
    direction LR

    subgraph Client/External Systems
        A[Web Frontend/Admin Panel] -->|HTTP/REST| FlaskApp
        B[Microsoft Identity Platform] -->|OAuth 2.0| OutlookSyncModule
    end

    subgraph Flask Main Application (app.py)
        subgraph Core API Endpoints
            API_FACE_UPLOAD[/api/face/upload]
            API_FACE_DELETE[/api/face/delete]
            API_FACE_CHECK[/api/face/check_user]
            API_FACE_RECOGNIZE[/api/face/recognize]
            API_FACE_TRIGGER[/api/face/trigger]
            API_LOG_FACE[/api/log/face]
            API_OUTLOOK_AUTH[/api/outlook/auth]
            API_OUTLOOK_CALLBACK[/api/outlook/callback]
            API_OUTLOOK_SYNC[/api/outlook/sync]
            API_FSM_START[/api/fsm/start]
            API_FSM_STATUS[/api/fsm/status]
            API_FSM_STOP[/api/fsm/stop]
            API_PEOPLE_UPDATE_STATUS[/api/people/update_status]
            API_WARNING[/api/warning]
        end

        subgraph Face Recognition Module
            FR_API(face_api.py)
            FR_REC_API(recognize_api.py)
            FR_LOG_API(log_api.py)
            FR_TRIG_API(trigger_api.py)
            FR_UTILS(face_utils.py)

            FR_API --- API_FACE_UPLOAD
            FR_API --- API_FACE_DELETE
            FR_API --- API_FACE_CHECK

            FR_REC_API --- API_FACE_RECOGNIZE
            FR_REC_API ---|Triggers FSM Update| API_PEOPLE_UPDATE_STATUS
            FR_REC_API -->|Video Stream| CAM_FACE(RTSP Camera: RTSP_URL_FACE)
            FR_REC_API -->|Uses| YOLO_FR(YOLO for Face Detection)
            FR_REC_API --> FR_UTILS
            FR_REC_API --> UL_LOG(utils.log_utils)
            FR_REC_API --> DB_STAFF(PostgreSQL: staff)
            FR_REC_API --> DB_CAL(PostgreSQL: calendar_event)

            FR_LOG_API --- API_LOG_FACE
            FR_LOG_API --> UL_LOG

            FR_TRIG_API --- API_FACE_TRIGGER
            FR_TRIG_API --> FR_REC_API
            FR_TRIG_API --> API_PEOPLE_UPDATE_STATUS

            FR_UTILS --> FS_FACE_LIB(File System: FACE_LIB_DIR)
        end

        subgraph Outlook Sync Module
            OS_AUTH_API(outlook_auth_api.py)
            OS_SYNC_API(outlook_sync_api.py)
            OS_DB_INIT(db_init_outlook.py)

            OS_AUTH_API --- API_OUTLOOK_AUTH
            OS_AUTH_API --- API_OUTLOOK_CALLBACK
            API_OUTLOOK_AUTH --> B
            API_OUTLOOK_CALLBACK --> B
            OS_SYNC_API --- API_OUTLOOK_SYNC
            OS_SYNC_API --> MS_GRAPH_API(Microsoft Graph API)
            OS_SYNC_API --> DB_CAL
            OS_SYNC_API --> REGEX_PARSE(Regex: 'Tour Guide' extraction)
        end

        subgraph People Counter Module
            PC_FSM_API(fsm_api.py)
            PC_FSM_RUNNER(fsm_runner.py Thread)

            PC_FSM_API --- API_FSM_START
            PC_FSM_API --- API_FSM_STATUS
            PC_FSM_API --- API_FSM_STOP
            PC_FSM_API --- API_PEOPLE_UPDATE_STATUS

            PC_FSM_API --> PC_FSM_RUNNER
            PC_FSM_RUNNER -->|Video Stream| CAM_MAIN(RTSP Camera: RTSP_URL)
            PC_FSM_RUNNER --> YOLO_PC(YOLO for Person Detection)
            PC_FSM_RUNNER --> BYTE_TRACKER(BYTETracker)
            PC_FSM_RUNNER -->|Triggers Face Recognition| API_FACE_TRIGGER
            PC_FSM_RUNNER -->|Triggers Warning| API_WARNING
            PC_FSM_RUNNER --> DB_PC(PostgreSQL: people_counter_state)
        end

        subgraph Warning Module
            WARN_API(warning_api.py)
            WARN_API --- API_WARNING
        end

        subgraph Utility Layer
            UL_DB(utils.db_utils.py)
            UL_LOG(utils.log_utils.py)

            UL_DB --> PostgresDB(PostgreSQL Database)
            UL_LOG --> UL_DB
        end
    end

    PostgresDB --- DB_STAFF
    PostgresDB --- DB_CAL
    PostgresDB --- DB_LOG(PostgreSQL: face_log)
    PostgresDB --- DB_PC
