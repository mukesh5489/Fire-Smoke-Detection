import argparse
import json
import os
import smtplib
import ssl
import sys
import time
import urllib.request
from datetime import datetime
from email.message import EmailMessage

import cv2
from ultralytics import YOLO

TARGET_LABELS = {"fire", "smoke"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fire and smoke detection and send email alerts when detected."
    )
    parser.add_argument("--source", default="rtsp://admin:MngrRWP1@192.168.1.48:554/Streaming/Channels/102",
                        help="Video source: 0 for webcam, a file path, or an RTSP stream URL.")
    parser.add_argument("--weights", default="weights/best.pt",
                        help="Path to the YOLO weights file.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Minimum confidence threshold for alerts.")
    parser.add_argument("--cooldown", type=int, default=120,
                        help="Seconds to wait between alert emails.")
    parser.add_argument("--output-dir", default="alerts",
                        help="Directory to save alert photos.")
    parser.add_argument("--smtp-server", default=os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
                        help="SMTP server host.")
    parser.add_argument("--smtp-port", type=int, default=int(os.environ.get("SMTP_PORT", 587)),
                        help="SMTP server port.")
    parser.add_argument("--smtp-user", default=os.environ.get("SMTP_USER", "tanuja9502190765@gmail.com"),
                        help="SMTP login username.")
    parser.add_argument("--smtp-password", default=os.environ.get("SMTP_PASSWORD", "wsom vyxq tkzt fkll"),
                        help="SMTP login password.")
    parser.add_argument("--from-email", default=os.environ.get("FROM_EMAIL", "tanuja9502190765@gmail.com"),
                        help="Sender email address.")
    parser.add_argument("--to-email", default=os.environ.get("TO_EMAIL", "mukesh19222326@gmail.com"),
                        help="Recipient email address.")
    parser.add_argument("--latitude", type=float, default=None,
                        help="Optional override latitude for accurate location.")
    parser.add_argument("--longitude", type=float, default=None,
                        help="Optional override longitude for accurate location.")
    parser.add_argument("--location-name", default=os.environ.get("LOCATION_NAME"),
                        help="Optional descriptive location text for manual coordinates.")
    parser.add_argument("--show", action="store_true",
                        help="Show the video stream with detections.")
    parser.add_argument("--no-location", action="store_true",
                        help="Do not attempt to resolve approximate location from IP.")
    return parser.parse_args()


def validate_config(args):
    missing = []
    for name in ["smtp_server", "smtp_user", "smtp_password", "from_email", "to_email"]:
        if getattr(args, name) is None:
            missing.append(name)
    if missing:
        print("Missing SMTP configuration:", ", ".join(missing))
        print("Set them on the command line or via environment variables: SMTP_SERVER, SMTP_USER, SMTP_PASSWORD, FROM_EMAIL, TO_EMAIL")
        sys.exit(1)


def get_location():
    url = "http://ip-api.com/json/"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.load(response)
    except Exception:
        return None

    if data.get("status") != "success":
        return None

    return {
        "latitude": data.get("lat"),
        "longitude": data.get("lon"),
        "city": data.get("city"),
        "region": data.get("regionName"),
        "country": data.get("country"),
        "zip": data.get("zip"),
        "ip": data.get("query"),
    }


def location_text(location):
    if not location:
        return "Location not available."
    if location.get("ip") == "manual":
        address = location.get("city") or "Manual location override"
        source = "manual"
    else:
        address = ", ".join(filter(None, [location.get("city"), location.get("region"), location.get("country")]))
        source = location.get("ip")
    return (
        f"Location: {address}\n"
        f"Latitude: {location['latitude']}, Longitude: {location['longitude']}\n"
        f"IP address: {source}"
    )


def save_alert_image(frame, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"alert_{timestamp}.jpg")
    cv2.imwrite(output_path, frame)
    return output_path


def build_email_body(alerts, location, image_path, source):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_lines = [
        "Dear Sir\n",
        "This is to inform you that fire and smoke have been detected in the Particular Area/Location at approximately " + timestamp + ".\n",
        "The situation was observed during routine monitoring, and immediate safety measures have been initiated. Relevant personnel have been notified, and necessary actions are being taken to assess and control the situation.\n",
        "Kindly treat this matter as urgent and advise if any further actions are required.\n",
        "Thank you.\n",
        "Regards,\n",
        "IT DEPARTMENT\n",
        "",
    ]
    if location:
        message_lines.extend(["Incident location details:\n", location_text(location), ""])
    else:
        message_lines.extend(["Incident location details:\n", "Location: unavailable (network or geolocation service failed).", ""])
    message_lines.append(f"Saved image: {image_path}")
    return "\n".join(message_lines)


def send_email(args, subject, body, attachment_path):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = args.from_email
    msg["To"] = args.to_email
    msg.set_content(body)

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            image_data = f.read()
        import imghdr

        image_type = imghdr.what(None, image_data) or "jpeg"
        msg.add_attachment(
            image_data,
            maintype="image",
            subtype=image_type,
            filename=os.path.basename(attachment_path),
        )

    context = ssl.create_default_context()
    with smtplib.SMTP(args.smtp_server, args.smtp_port, timeout=30) as smtp:
        smtp.starttls(context=context)
        smtp.login(args.smtp_user, args.smtp_password)
        smtp.send_message(msg)


def get_alerts(results, class_names, confidence_threshold):
    alerts = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        classes = [int(x) for x in boxes.cls] if hasattr(boxes, "cls") else []
        confidences = [float(x) for x in boxes.conf] if hasattr(boxes, "conf") else []

        for cls, conf in zip(classes, confidences):
            label = class_names.get(cls, str(cls)).lower()
            if label in TARGET_LABELS and conf >= confidence_threshold:
                alerts.append((label, conf))
    return alerts


def process_frame(frame, model, args, location, last_alert_time):
    results = model(frame, conf=args.conf)
    alerts = get_alerts(results, model.names, args.conf)
    if alerts:
        now = time.time()
        if now - last_alert_time >= args.cooldown:
            image_path = save_alert_image(frame, args.output_dir)
            body = build_email_body(alerts, location, image_path, args.source)
            subject = "Fire/Smoke Alert Detected"
            try:
                send_email(args, subject, body, image_path)
                print(f"Alert sent and image saved to {image_path}")
            except Exception as exc:
                print("Failed to send email alert:", exc)
            last_alert_time = now
    return results, last_alert_time


def main():
    args = parse_args()
    validate_config(args)

    if args.latitude is not None and args.longitude is not None:
        location = {
            "latitude": args.latitude,
            "longitude": args.longitude,
            "city": args.location_name,
            "region": None,
            "country": None,
            "zip": None,
            "ip": "manual",
        }
    elif not args.no_location:
        location = get_location()
    else:
        location = None

    try:
        model = YOLO(args.weights)
    except Exception as exc:
        print("Unable to load model weights:", exc)
        sys.exit(1)

    source = args.source
    if source.isdigit():
        source = int(source)

    if isinstance(source, str) and os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            frame = cv2.imread(source)
            if frame is None:
                print(f"Unable to read image file: {source}")
                sys.exit(1)
            results, _ = process_frame(frame, model, args, location, last_alert_time=0)
            if args.show:
                annotated = results[0].plot() if len(results) > 0 else frame
                cv2.imshow("Fire/Smoke Detection", annotated)
                cv2.waitKey(0)
            return

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Unable to open video source: {args.source}")
        sys.exit(1)

    last_alert_time = 0
    print("Starting detection. Press ESC to quit.")
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        results, last_alert_time = process_frame(frame, model, args, location, last_alert_time)

        if args.show:
            annotated = results[0].plot() if len(results) > 0 else frame
            cv2.imshow("Fire/Smoke Detection", annotated)
            if cv2.waitKey(1) == 27:  # ESC key
                break

    capture.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()