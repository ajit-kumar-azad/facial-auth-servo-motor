import io
import time
from typing import List

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
import face_recognition

# ==========================
# CONFIG
# ==========================

API_BASE_URL = "http://127.0.0.1:8000"  # FastAPI server


# ==========================
# HELPER FUNCTIONS
# ==========================

import base64
import serial
import serial.tools.list_ports
import time

def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

def connect_serial(port: str, baud: int = 9600, timeout: float = 1.0):
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        # tiny settle
        time.sleep(0.3)
        # clear any boot text
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        return ser
    except Exception as e:
        st.error(f"Failed to open serial {port}: {e}")
        return None

def send_cmd(ser: serial.Serial | None, cmd: str, expect_ok: bool = False, show_feedback: bool = True):
    if ser is None or not ser.is_open:
        if show_feedback:
            st.error("Serial not connected.")
        return False, ""
    try:
        ser.write((cmd.strip() + "\n").encode("utf-8"))
        ser.flush()
        if expect_ok:
            line = ser.readline().decode(errors="ignore").strip()
            if show_feedback:
                st.caption(f"Arduino: {line}")
            return ("OK" in line or "READY" in line), line
        return True, ""
    except Exception as e:
        if show_feedback:
            st.error(f"Serial write failed: {e}")
        return False, ""

def play_audio_autoplay(mp3_bytes: bytes):
    b64 = base64.b64encode(mp3_bytes).decode("ascii")
    audio_html = f"""
    <audio autoplay style="display:none">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def api_create_user(name: str, image_bytes: bytes):
    files = {
        "image": ("image.jpg", image_bytes, "image/jpeg"),
    }
    data = {
        "name": name,
    }
    resp = requests.post(f"{API_BASE_URL}/users", data=data, files=files)
    return resp


def api_list_users(limit: int = 50, offset: int = 0):
    resp = requests.get(f"{API_BASE_URL}/users", params={"limit": limit, "offset": offset})
    resp.raise_for_status()
    return resp.json()


def api_delete_user(user_id: int):
    resp = requests.delete(f"{API_BASE_URL}/users/{user_id}")
    return resp


def api_identify(image_bytes: bytes):
    files = {
        "image": ("image.jpg", image_bytes, "image/jpeg"),
    }
    resp = requests.post(f"{API_BASE_URL}/identify", files=files)
    return resp


def api_list_users_with_embeddings():
    """
    OPTIONAL: if you add an endpoint /users_full that returns embeddings too,
    you can use this. For now, we'll assume you added:

    @app.get("/users_full", response_model=List[UserWithEmbedding])
    """
    resp = requests.get(f"{API_BASE_URL}/users_full")
    resp.raise_for_status()
    return resp.json()


def load_known_faces_from_api():
    """
    Load known faces (embeddings + metadata) for live camera recognition.
    Requires /users_full endpoint (id, name, embedding[]).
    """
    try:
        users = api_list_users_with_embeddings()
    except Exception as e:
        st.error(f"Error loading users with embeddings: {e}")
        return [], [], []

    encodings = []
    names = []
    ids = []
    for u in users:
        if "embedding" in u and u["embedding"]:
            encodings.append(np.array(u["embedding"], dtype=np.float32))
            names.append(u["name"])
            ids.append(u["id"])
    return encodings, names, ids


def draw_labeled_boxes(frame_bgr, face_locations, face_names, colors):
    """
    Draw rectangles and labels on the frame.
    """
    for (top, right, bottom, left), name, color in zip(face_locations, face_names, colors):
        # Rectangle
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

        # Label background
        label = name
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1
        )
        cv2.rectangle(
            frame_bgr,
            (left, bottom + baseline),
            (left + text_width, bottom + text_height + 2 * baseline),
            color,
            cv2.FILLED,
        )

        # Label text
        cv2.putText(
            frame_bgr,
            label,
            (left, bottom + text_height + baseline),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    return frame_bgr


# ==========================
# STREAMLIT UI
# ==========================

st.set_page_config(
    page_title="Face Recognition Manager",
    page_icon="😎",
    layout="wide",
)

st.title("😎 Face Recognition Manager")
st.caption("Manage users & live face recognition using your FastAPI backend")


# Sidebar config
st.sidebar.header("Settings")
api_url_input = st.sidebar.text_input("API Base URL", value=API_BASE_URL)
if api_url_input:
    API_BASE_URL = api_url_input.rstrip("/")

st.sidebar.markdown("---")
st.sidebar.write("Backend status:")
try:
    # quick ping by calling GET /users with limit=1
    _ = requests.get(f"{API_BASE_URL}/users", params={"limit": 1}, timeout=2)
    st.sidebar.success("✅ Connected")
except Exception as e:
    st.sidebar.error("❌ Cannot reach API")
    st.sidebar.code(str(e))

st.sidebar.markdown("### Arduino")
available_ports = list_serial_ports()
port = st.sidebar.selectbox("Serial port", options=[""] + available_ports, index=0, help="e.g., /dev/ttyACM0")
baud = st.sidebar.number_input("Baud", value=9600, step=1)

if "arduino" not in st.session_state:
    st.session_state["arduino"] = None

colA, colB = st.sidebar.columns(2)
if colA.button("Connect"):
    if not port:
        st.sidebar.error("Select a serial port.")
    else:
        ser = connect_serial(port, baud)
        if ser:
            st.sidebar.success("Connected")
            st.session_state["arduino"] = ser
if colB.button("Disconnect"):
    ser = st.session_state.get("arduino")
    if ser and ser.is_open:
        ser.close()
    st.session_state["arduino"] = None
    st.sidebar.info("Disconnected")

# Quick test buttons
t1, t2 = st.sidebar.columns(2)
if t1.button("Test OPEN"):
    send_cmd(st.session_state.get("arduino"), "SERVO 180", expect_ok=True)
if t2.button("Test CLOSE"):
    send_cmd(st.session_state.get("arduino"), "SERVO 0", expect_ok=True)


tab1, tab2, tab3 = st.tabs(["👤 User Management", "🖼 Identify from Photo", "📷 Live Camera Recognition"])


# -------------------------------------------
# TAB 1: USER MANAGEMENT (Add New User)
# -------------------------------------------
with tab1:
    st.subheader("Add New User")

    col1, col2 = st.columns([1, 2])

    with col1:
        new_name = st.text_input("Name", key="add_name")

        source = st.radio(
            "Photo source",
            ["Upload", "Use Webcam"],
            horizontal=True,
            key="add_photo_source",
        )

        uploaded_image = None
        camera_photo = None

        if source == "Upload":
            uploaded_image = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"], key="uploader_add")
        else:
            # st.camera_input returns an UploadedFile-like object
            camera_photo = st.camera_input("Take a photo", key="camera_add")

        # choose the active image bytes
        image_bytes = None
        if uploaded_image:
            # .getvalue() works for both file_uploader & camera_input
            image_bytes = uploaded_image.getvalue()
        elif camera_photo:
            image_bytes = camera_photo.getvalue()

        if st.button("Add User", type="primary", key="btn_add_user"):
            if not new_name:
                st.warning("Please enter a name.")
            elif not image_bytes:
                st.warning("Please provide a photo (upload or webcam).")
            else:
                with st.spinner("Creating user..."):
                    resp = api_create_user(new_name, image_bytes)
                if resp.status_code == 200:
                    st.success("User created successfully!")
                    st.json(resp.json())
                else:
                    st.error(f"Error: {resp.status_code}")
                    try:
                        st.json(resp.json())
                    except Exception:
                        st.text(resp.text)

    with col2:
        st.markdown("#### Preview")
        if uploaded_image:
            st.image(uploaded_image, width=320, caption="Uploaded image")
        elif camera_photo:
            st.image(camera_photo, width=320, caption="Captured image")

    st.markdown("---")
    st.subheader("View & Delete Users")

    list_limit = st.number_input("Number of users to fetch", 1, 500, 50, key="list_limit")
    if st.button("Refresh User List", key="btn_refresh_list"):
        st.session_state["users_list"] = api_list_users(limit=list_limit)

    users_list = st.session_state.get("users_list", [])
    if users_list:
        for u in users_list:
            with st.container(border=True):
                cols = st.columns([1, 1, 3, 1])
                cols[0].markdown(f"**ID:** {u['id']}")
                cols[1].image(u["photo_url"], width=72)
                cols[2].markdown(f"**Name:** {u['name']}")
                if cols[3].button("Delete", key=f"delete_{u['id']}"):
                    resp = api_delete_user(u["id"])
                    if resp.status_code == 200:
                        st.success(f"Deleted user {u['id']}")
                        users_list = [x for x in users_list if x["id"] != u["id"]]
                        st.session_state["users_list"] = users_list
                    else:
                        st.error(f"Error deleting user {u['id']}: {resp.text}")
    else:
        st.info("No users loaded yet. Click **Refresh User List**.")



# ===========================================
# TAB 2: IDENTIFY FROM PHOTO (API /identify)
# ===========================================

with tab2:
    st.subheader("Identify a User from a Photo")

    col_left, col_right = st.columns(2)

    with col_left:
        identify_image = st.file_uploader(
            "Upload an image to identify", type=["jpg", "jpeg", "png"], key="identify_uploader"
        )
        if st.button("Identify", type="primary"):
            if not identify_image:
                st.warning("Please upload an image.")
            else:
                image_bytes = identify_image.read()
                with st.spinner("Calling /identify API..."):
                    resp = api_identify(image_bytes)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(f"Result: **{result.get('name', 'UNKNOWN')}**")
                    st.json(result)
                else:
                    st.error(f"Error: {resp.status_code}")
                    try:
                        st.json(resp.json())
                    except Exception:
                        st.write(resp.text)

    with col_right:
        st.markdown("#### Image Preview")
        if identify_image:
            st.image(identify_image, caption="Image to identify", use_container_width=True)


# ===========================================
# TAB 3: LIVE CAMERA RECOGNITION (Speak & Stop)
# ===========================================

from gtts import gTTS
import io

with tab3:
    st.subheader("Live Camera Recognition (with Servo Trigger)")
    st.markdown(
        "When a **known** face is detected, send **OPEN**, wait 2s, then **CLOSE** to the Arduino servo. "
        "Detection continues afterwards."
    )

    if st.button("Load known faces from API", key="btn_load_known"):
        with st.spinner("Loading known embeddings..."):
            encodings, names, ids = load_known_faces_from_api()
        if encodings:
            st.success(f"Loaded {len(encodings)} users with embeddings.")
            st.session_state["known_encodings"] = encodings
            st.session_state["known_names"] = names
            st.session_state["known_ids"] = ids
        else:
            st.warning("No embeddings found. Make sure /users_full is implemented and users exist.")

    known_encodings = st.session_state.get("known_encodings", [])
    known_names = st.session_state.get("known_names", [])

    # state vars
    st.session_state.setdefault("start_cam_servo", False)
    st.session_state.setdefault("servo_cooldown_until", 0.0)  # epoch seconds to prevent rapid retriggers
    start_cam_servo = st.checkbox("Start Camera", value=st.session_state["start_cam_servo"], key="start_cam_servo")

    FRAME_PLACEHOLDER = st.empty()

    if start_cam_servo:
        if not known_encodings:
            st.warning("No known faces loaded. Click **Load known faces from API** first.")
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Check your camera permissions.")
            else:
                st.info("Streaming… known face will trigger the servo (open→2s→close).")
                FACE_MATCH_THRESHOLD = 0.6

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame from camera.")
                        break

                    # RGB contiguous frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

                    face_locations = face_recognition.face_locations(rgb_frame)
                    try:
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    except TypeError:
                        face_encodings = face_recognition.face_encodings(rgb_frame)

                    names_for_frame, colors_for_frame = [], []
                    authorized_detected = False
                    authorized_name = None

                    for face_encoding in face_encodings:
                        distances = np.linalg.norm(np.array(known_encodings) - face_encoding, axis=1)
                        if len(distances) == 0:
                            cur_name, cur_color = "UNKNOWN", (0, 0, 255)
                        else:
                            best_idx = int(np.argmin(distances))
                            best_dist = float(distances[best_idx])
                            if best_dist <= FACE_MATCH_THRESHOLD:
                                cur_name, cur_color = known_names[best_idx], (0, 255, 0)
                                authorized_detected = True
                                authorized_name = cur_name
                            else:
                                cur_name, cur_color = "UNKNOWN", (0, 0, 255)

                        names_for_frame.append(cur_name)
                        colors_for_frame.append(cur_color)

                    frame_with_boxes = draw_labeled_boxes(frame, face_locations, names_for_frame, colors_for_frame)
                    FRAME_PLACEHOLDER.image(frame_with_boxes[:, :, ::-1], channels="RGB", use_container_width=True)

                    # if we have any detection: speak and end
                    if authorized_name is not None:
                        detected_name_once = authorized_name
                        # TTS -> autoplay
                        try:
                            buf = io.BytesIO()
                            text_to_say = detected_name_once if detected_name_once != "UNKNOWN" else "Unknown"
                            gTTS(text=text_to_say, lang="en").write_to_fp(buf)
                            buf.seek(0)
                            play_audio_autoplay(buf.read())
                        except Exception as tts_err:
                            st.warning(f"Could not synthesize audio: {tts_err}")

                    # Trigger servo if authorized face and cooldown passed
                    now = time.time()
                    if authorized_detected and now >= st.session_state["servo_cooldown_until"]:
                        ser = st.session_state.get("arduino")
                        if ser is None or not ser.is_open:
                            st.warning("Authorized face detected but Arduino is not connected.")
                        else:
                            st.success(f"Authorized: {authorized_name or 'User'} → OPEN door")
                            ok, _ = send_cmd(ser, "SERVO 180", expect_ok=True, show_feedback=False)
                            if not ok:
                                st.error("Failed to send OPEN")
                            # door stays open for 2 seconds
                            time.sleep(2.0)
                            ok, _ = send_cmd(ser, "SERVO 0", expect_ok=True, show_feedback=False)
                            if not ok:
                                st.error("Failed to send CLOSE")
                            # small cooldown to avoid re-triggering every frame
                            st.session_state["servo_cooldown_until"] = time.time() + 3.0

                    # keep the loop running smoothly
                    time.sleep(0.03)

                cap.release()
                FRAME_PLACEHOLDER.empty()
    else:
        st.info("Enable **Start Camera** to begin live recognition.")
