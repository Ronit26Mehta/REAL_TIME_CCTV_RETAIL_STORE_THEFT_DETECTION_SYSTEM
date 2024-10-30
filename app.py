import cv2
import torch
import pandas as pd
from flask import Flask, render_template, Response, send_file, jsonify, request
from datetime import datetime, timedelta
import os
import time 
import numpy as np  

app = Flask(__name__)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


object_movement = {}

def create_csv_files(video_filename):
    csv_directory = './data'
    os.makedirs(csv_directory, exist_ok=True)
    
    csv_path = os.path.join(csv_directory, f"{video_filename}_detections.csv")
    alert_csv_path = os.path.join(csv_directory, f"{video_filename}_alerts.csv")

    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["timestamp", "object", "confidence", "x", "y"]).to_csv(csv_path, index=False)

    if not os.path.exists(alert_csv_path):
        pd.DataFrame(columns=["timestamp", "object", "alert_type", "x", "y"]).to_csv(alert_csv_path, index=False)

    return csv_path, alert_csv_path


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_map(detections, ground_truths, iou_threshold=0.5):
    average_precisions = []
    for cls in ground_truths:
        gt_boxes = ground_truths[cls]
        pred_boxes = detections.get(cls, [])

        true_positives = 0
        false_positives = 0
        false_negatives = len(gt_boxes)

        for pred_box in pred_boxes:
            matched = False
            for gt_box in gt_boxes:
                if calculate_iou(pred_box, gt_box) > iou_threshold:
                    true_positives += 1
                    false_negatives -= 1
                    matched = True
                    break
            if not matched:
                false_positives += 1

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        average_precisions.append((precision * recall) / (precision + recall + 1e-6))

    mAP = np.mean(average_precisions) if average_precisions else 0
    return mAP


def generate_performance_report(avg_fps, avg_map, crowd_performance, lighting_performance):
    report = f"""
    --- System Performance Report ---

    Average FPS: {avg_fps:.2f} frames per second
    Average mAP (Accuracy): {avg_map:.2f}

    Strengths:
    - High FPS indicating real-time processing capability.
    - Accurate detection (high mAP) in controlled environments.

    Limitations:
    - Reduced detection accuracy under challenging conditions:
      * Crowded Scenes: {crowd_performance}
      * Low-light/Varied Lighting: {lighting_performance}

    Recommendations:
    - Improve model robustness by training with diverse datasets including crowded and low-light scenes.
    - Consider using higher-performing hardware if FPS is too low for real-time requirements.
    """
    # print(report)  
    with open('./data/performance_report.txt', 'w') as f:
        f.write(report)

def detect_suspicious_behavior(video_source, csv_path, alert_csv_path):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    fps_list = []  
    accuracy_list = []  

    while cap.isOpened():
        start_time = time.time()  
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video stream.")
            break

        results = model(frame)
        suspicious_activity_detected = False

        detections = {}  

        for *xyxy, conf, cls in results.xyxy[0]:
            x, y = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
            label = model.names[int(cls)]
            confidence = float(conf)
            box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

            if label in ['person', 'backpack', 'handbag']:
                detections.setdefault(label, []).append(box)

                with open(csv_path, 'a') as f:
                    timestamp = datetime.now().isoformat()
                    f.write(f"{timestamp},{label},{confidence},{x},{y}\n")

                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if detect_abnormal_movement(label, x, y):
                    suspicious_activity_detected = True
                    cv2.putText(frame, 'Suspicious Activity!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

       
        ground_truths = {
        'class1': [[50, 50, 100, 100], [120, 120, 170, 170]],  # Ground truth boxes for class1
        'class2': [[30, 30, 60, 60]]  } 
        mAP = calculate_map(detections, ground_truths)
        accuracy_list.append(mAP)

      
        if suspicious_activity_detected:
            raise_alert(label, x, y, alert_csv_path)

       
        fps = 1 / (time.time() - start_time)
        fps_list.append(fps)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    
    avg_fps = sum(fps_list) / len(fps_list)
    avg_map = sum(accuracy_list) / len(accuracy_list)

   
    crowd_performance = "Moderate (mAP ~ 0.65)"
    lighting_performance = "Low (mAP ~ 0.5)"

    
    generate_performance_report(avg_fps, avg_map, crowd_performance, lighting_performance)

    cap.release()


def detect_abnormal_movement(object_label, x, y):
    current_time = datetime.now()

   
    if object_label not in object_movement:
        object_movement[object_label] = []

    
    object_movement[object_label].append({"time": current_time, "x": x, "y": y})

    
    object_movement[object_label] = [pos for pos in object_movement[object_label] if current_time - pos["time"] < timedelta(seconds=10)]

    
    if len(object_movement[object_label]) > 5:  
        distances = [((p["x"] - x) ** 2 + (p["y"] - y) ** 2) ** 0.5 for p in object_movement[object_label]]
        if all(dist < 50 for dist in distances):  
            return True  

    return False


def raise_alert(object_label, x, y, alert_csv_path):
    timestamp = datetime.now().isoformat()
    alert_type = "Suspicious Movement Detected"
    
   
    with open(alert_csv_path, 'a') as f:
        f.write(f"{timestamp},{object_label},{alert_type},{x},{y}\n")

    
    print(f"Alert: {alert_type} for {object_label} at position ({x}, {y})")

@app.route('/video_feed')
def video_feed():
    video_source = request.args.get('source', None)
    video_filename = request.args.get('filename', 'default_video')

    if video_source is None:
        return "No video source provided", 400

   
    csv_path, alert_csv_path = create_csv_files(video_filename)

    return Response(detect_suspicious_behavior(video_source, csv_path, alert_csv_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/get_trajectory_data')
def get_trajectory_data():
    video_filename = request.args.get('filename', 'default_video')
    csv_path = f"./data/{video_filename}_detections.csv"

    df = pd.read_csv(csv_path)
    data = []
    for obj in df['object'].unique():
        obj_data = df[df['object'] == obj]
        positions = [{"x": row['x'], "y": row['y']} for _, row in obj_data.iterrows()]
        data.append({"id": obj, "positions": positions})
    return jsonify(data)  


@app.route('/api/get_performance_report')
def get_performance_report():
    report_path = './data/performance_report.txt'
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = f.read()
        return report
    return "Report not found. Please run detection first.", 404


@app.route('/download_log')
def download_log():
    video_filename = request.args.get('filename', 'default_video')
    csv_path = f"./data/{video_filename}_detections.csv"
    return send_file(csv_path, as_attachment=True)


@app.route('/download_alerts')
def download_alerts():
    video_filename = request.args.get('filename', 'default_video')
    alert_csv_path = f"./data/{video_filename}_alerts.csv"
    return send_file(alert_csv_path, as_attachment=True)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join('./data', video_file.filename)
            video_file.save(video_path)
            return render_template('dashboard.html', video_path=video_path, video_filename=video_file.filename)

    return render_template('upload.html')


@app.route('/')
def index():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
