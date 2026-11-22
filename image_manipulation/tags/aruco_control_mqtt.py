"""
PC side: ArUco tracking + keyboard commands + MQTT motion feedback.

Dependencies:
    pip install opencv-contrib-python numpy paho-mqtt

MQTT configuration follows mqtt_input_img.py:
    Broker: broker.hivemq.com (default)
    Port:   1883 (default)
    Keepalive: 60 seconds
    Callback API: VERSION2

The PC publishes JSON messages on:
    motion/state
    motion/feedback

Configure FLOOR_TAGS, ROBOT_TAG_ID, CAMERA_INDEX and HEADING_OFFSET_DEG
for your physical installation.
"""

import cv2
import numpy as np
import json
import math
import time
import threading
import paho.mqtt.client as mqtt


# ---------------- USER CONFIGURATION ----------------

CAMERA_INDEX = 1

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ROBOT_TAG_ID = 20

# EXAMPLE ONLY: replace IDs and coordinates with your real floor layout.
# Coordinates are the CENTER of each floor marker, in decimeters.
# At least four non-collinear floor markers must be visible.
FLOOR_TAGS = {
    0: (0.0, 0.0),
    1: (4.5, 0.0),
    2: (4.5, 4.5),
    3: (0.0, 4.5),
}

# If marker edge 0->1 does not point toward robot-forward, adjust this.
HEADING_OFFSET_DEG = 0.0

# MQTT topics
PREFIX = "UDFJC/carrobot/robot30/"
STATE_TOPIC = PREFIX + "motion/state"
FEEDBACK_TOPIC = PREFIX + "motion/feedback"

# Keyboard commands (latched: press a key to select a movement; SPACE/X stops)
V_COMMAND_DM_S = 5.0
W_COMMAND_DEG_S = 45.0

COMMAND_REFRESH_S = 0.15
COMMAND_LEASE_S = 0.45
FEEDBACK_PERIOD_S = 0.10

# ----------------------------------------------------


def on_connect(client, userdata, flags, reason_code, properties):
    """Paho MQTT Callback API VERSION2 connection callback."""
    if reason_code == 0:
        print("\\nConnected successfully to MQTT broker.")
    else:
        print(f"\\nMQTT connection error. Code: {reason_code}")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    if reason_code != 0:
        print(f"MQTT disconnected unexpectedly. Code: {reason_code}")


def publish_json(client, topic, data):
    """Serialize a dictionary as JSON and publish it to an MQTT topic."""
    payload = json.dumps(data, separators=(",", ":"))
    info = client.publish(topic, payload)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        print(f"Could not publish to {topic}. MQTT error: {info.rc}")


def homography_from_floor(corners, ids):
    """Estimate image-pixel -> floor-coordinate homography."""
    image_points = []
    world_points = []

    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):
        marker_id = int(marker_id)
        if marker_id in FLOOR_TAGS:
            center_px = np.asarray(corners[i][0], dtype=np.float32).mean(axis=0)
            image_points.append(center_px)
            world_points.append(FLOOR_TAGS[marker_id])

    if len(image_points) < 4:
        return None

    H, _ = cv2.findHomography(
        np.asarray(image_points, dtype=np.float32),
        np.asarray(world_points, dtype=np.float32),
        cv2.RANSAC,
        0.5,  # reprojection threshold in dm
    )
    return H


def robot_pose_from_marker(corners, ids, H):
    """Return robot pose (x dm, y dm, theta radians), or None."""
    if ids is None or H is None:
        return None

    found = np.where(ids.flatten() == ROBOT_TAG_ID)[0]
    if len(found) == 0:
        return None

    marker_corners = np.asarray(corners[int(found[0])][0], dtype=np.float32)

    # Marker center and midpoint of edge 0->1.
    # Mount the marker so edge 0->1 points toward robot-forward.
    center_px = marker_corners.mean(axis=0)
    front_px = (marker_corners[0] + marker_corners[1]) / 2.0

    image_points = np.asarray(
        [center_px, front_px], dtype=np.float32
    ).reshape(-1, 1, 2)

    world_points = cv2.perspectiveTransform(image_points, H).reshape(-1, 2)

    x, y = float(world_points[0, 0]), float(world_points[0, 1])
    dx = float(world_points[1, 0] - world_points[0, 0])
    dy = float(world_points[1, 1] - world_points[0, 1])

    theta = math.atan2(dy, dx) + math.radians(HEADING_OFFSET_DEG)
    theta = math.atan2(math.sin(theta), math.cos(theta))
    return x, y, theta


def motion_between(pose0, pose1, dt):
    """Return signed forward speed [dm/s] and angular speed [deg/s]."""
    if dt <= 0:
        return None

    x0, y0, th0 = pose0
    x1, y1, th1 = pose1

    dx = x1 - x0
    dy = y1 - y0
    dtheta = math.atan2(math.sin(th1 - th0), math.cos(th1 - th0))
    theta_mid = th0 + 0.5 * dtheta

    # Signed displacement projected onto the robot's forward axis.
    v = (dx * math.cos(theta_mid) + dy * math.sin(theta_mid)) / dt
    w_deg_s = math.degrees(dtheta / dt)
    return v, w_deg_s


def main():
    # Same interactive defaults as mqtt_input_img.py.
    broker = input("Broker [broker.hivemq.com]: ") or "broker.hivemq.com"
    port = int(input("Puerto [1883]: ") or 1883)

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    print(f"Connecting to MQTT broker {broker}:{port} ...")
    mqtt_client.connect(broker, port, 60)
    mqtt_client.loop_start()

    cap = None
    seq = 0
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Check CAMERA_INDEX.")

        v_cmd = 0.0
        w_cmd = 0.0
        last_command_send = 0.0
        last_pose = None
        last_pose_time = None
        feedback_pose = None
        feedback_time = None
        skip_next_feedback = False

        print("Keys: W forward, S reverse, A left, D right, SPACE/X stop, Q quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read camera frame")
                break

            now = time.perf_counter()
            corners, ids, rejected = detector.detectMarkers(frame)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            H = homography_from_floor(corners, ids)
            pose = robot_pose_from_marker(corners, ids, H)

            # Keyboard is intentionally latched: a key selects a state.
            key = cv2.waitKey(1) & 0xFF
            new_command = None

            if key in (ord("w"), ord("W")):
                new_command = (V_COMMAND_DM_S, 0.0)
            elif key in (ord("s"), ord("S")):
                new_command = (-V_COMMAND_DM_S, 0.0)
            elif key in (ord("a"), ord("A")):
                new_command = (0.0, W_COMMAND_DEG_S)
            elif key in (ord("d"), ord("D")):
                new_command = (0.0, -W_COMMAND_DEG_S)
            elif key in (ord(" "), ord("x"), ord("X")):
                new_command = (0.0, 0.0)
            elif key in (ord("q"), ord("Q")):
                break

            if new_command is not None and new_command != (v_cmd, w_cmd):
                v_cmd, w_cmd = new_command
                seq += 1
                skip_next_feedback = True
                last_command_send = 0.0

            # Refresh state periodically; Pico should stop when its lease expires.
            if now - last_command_send >= COMMAND_REFRESH_S:
                command = {
                    "v_dm/s": v_cmd,
                    "w_deg/s": w_cmd,
                    "alpha0_deg": 0,
                    "alpha1_deg": 0,
                    "alpha2_deg": 0,
                    "duration_s": COMMAND_LEASE_S,
                    "seq": seq,
                }
                publish_json(mqtt_client, STATE_TOPIC, command)
                last_command_send = now

            if pose is not None:
                x, y, theta = pose

                if last_pose is not None and last_pose_time is not None:
                    dt_frame = now - last_pose_time
                    measured = motion_between(last_pose, pose, dt_frame)
                else:
                    measured = None

                # Publish measured motion periodically.
                if feedback_pose is None:
                    feedback_pose = pose
                    feedback_time = now
                elif now - feedback_time >= FEEDBACK_PERIOD_S:
                    dt_feedback = now - feedback_time
                    measured_feedback = motion_between(
                        feedback_pose, pose, dt_feedback
                    )

                    if measured_feedback is not None:
                        vm, wm = measured_feedback

                        if skip_next_feedback:
                            # This interval may include motion from two commands.
                            skip_next_feedback = False
                        else:
                            feedback = {
                                "v_dm/s": vm,
                                "w_deg/s": wm,
                                "duration_s": dt_feedback,
                                "seq": seq,
                            }
                            publish_json(mqtt_client, FEEDBACK_TOPIC, feedback)

                    feedback_pose = pose
                    feedback_time = now

                last_pose = pose
                last_pose_time = now

                cv2.putText(
                    frame, f"X={x:.2f} dm  Y={y:.2f} dm",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, f"theta={math.degrees(theta):.1f} deg",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA
                )
                if measured is not None:
                    cv2.putText(
                        frame,
                        f"v={measured[0]:.2f} dm/s  w={measured[1]:.1f} deg/s",
                        (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2, cv2.LINE_AA
                    )
            else:
                cv2.putText(
                    frame, "Need robot tag + 4 floor tags",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2, cv2.LINE_AA
                )
                # Do not calculate velocity across a tracking gap.
                last_pose = None
                last_pose_time = None
                feedback_pose = None
                feedback_time = None

            cv2.putText(
                frame,
                f"MQTT {broker}:{port} | CMD v={v_cmd:.1f} w={w_cmd:.1f} seq={seq}",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 0), 1, cv2.LINE_AA
            )

            cv2.imshow("CarroBot ArUco calibration - MQTT", frame)

    finally:
        # Publish a final stop request before disconnecting.
        try:
            stop_command = {
                "v_dm/s": 0.0,
                "w_deg/s": 0.0,
                "alpha0_deg": 0,
                "alpha1_deg": 0,
                "alpha2_deg": 0,
                "duration_s": 0.2,
                "seq": seq + 1,
            }
            stop_info = mqtt_client.publish(
                STATE_TOPIC, json.dumps(stop_command, separators=(",", ":"))
            )
            stop_info.wait_for_publish(timeout=1.0)
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        except Exception as exc:
            print("Error while closing MQTT:", exc)

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
