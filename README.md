# REAL-TIME CCTV FOOTAGE-BASED THEFT DETECTION SYSTEM

## demo video:



https://github.com/user-attachments/assets/a5488a07-7ea7-4d33-9383-17bb13d1653b




## Project Overview

This project is a real-time CCTV-based theft detection system designed to track and monitor suspicious activities using a combination of computer vision and deep learning. Built using Flask, OpenCV, and YOLOv5, the system processes live video feeds, detects and tracks objects, and flags suspicious movements as potential theft. Detected events are recorded in CSV files, and a web interface is available for real-time monitoring.

## Directory Structure

```
REAL-TIME_CCTV_FOOTAGE_BASED_THEFT_DETECTION_SYSTEM/
│
├── app.py                   # Main application file containing Flask routes and detection functions
├── yolov5s.pt               # YOLOv5 model file for object detection
├── data/                    # Directory for storing video files and generated CSV files
│   ├── [video_files].mp4    # Sample video files for testing
│   ├── video_stream_alerts.csv  # Alerts generated by suspicious behavior detection
│   ├── video_stream_detections.csv  # Object detection records for the video stream
│   ├── performance_report.txt  # System performance report file
│   └── your_video_name_detections.csv / your_video_name_alerts.csv
│                               # Custom CSVs based on uploaded video names
└── templates/               # HTML templates for the Flask web interface
    ├── dashboard.html       # Dashboard to view real-time footage and alerts
    └── upload.html          # Interface for uploading video files
```

## How the Project Works

1. **Object Detection Using YOLOv5**: The pre-trained YOLOv5 model (`yolov5s.pt`) is loaded to detect objects such as people and bags in each video frame.
2. **Real-Time Object Tracking**: Each detected object is monitored across frames to track movement. If abnormal movement patterns are identified, it triggers an alert.
3. **Data Storage**: Detections and alerts are saved as CSV files, which are automatically generated for each video processed.
4. **Performance Reporting**: The system tracks average FPS (frames per second) and mean Average Precision (mAP) for object detection accuracy under various conditions. A performance report is generated based on these metrics.
5. **Web Interface**: A Flask-based web application provides a dashboard to view real-time video feeds and alert logs. Users can also upload new video files.

## Main Functions in `app.py`

### `create_csv_files(video_filename)`
This function sets up CSV files in the `data/` directory for each video:
- **Detections CSV**: Records the timestamp, object type, confidence, and coordinates for each detected object.
- **Alerts CSV**: Logs suspicious activity with timestamps, object types, alert type, and coordinates.

### `calculate_iou(boxA, boxB)`
Calculates the Intersection over Union (IoU) of two bounding boxes. Used for tracking the same object across frames.

### `calculate_map(detections, ground_truths, iou_threshold=0.5)`
This function calculates the mean Average Precision (mAP) to assess detection accuracy. It compares predicted detections with ground truth data based on IoU.

### `generate_performance_report(avg_fps, avg_map, crowd_performance, lighting_performance)`
Generates a performance report summarizing the system’s average FPS, accuracy (mAP), and limitations under challenging conditions (e.g., crowded scenes, low-light settings).

### `detect_suspicious_behavior(video_source, csv_path, alert_csv_path)`
The main function that performs real-time object detection and monitors for abnormal movement patterns. If suspicious behavior is detected, it raises an alert and logs it.

### `detect_abnormal_movement(object_label, x, y)`
Tracks the movement of specific objects over time to identify abnormal patterns that may suggest suspicious behavior.

### `raise_alert(object_label, x, y, alert_csv_path)`
Logs alerts for suspicious activities by writing to the alert CSV file.

### Flask Routes

- **`/`**: Serves the upload page where users can upload video files.
- **`/upload`**: Handles video uploads and renders the `dashboard.html` page for monitoring.
- **`/video_feed`**: Streams the uploaded video feed with real-time detections and alerts.
- **`/api/get_trajectory_data`**: Returns object trajectory data from the detection CSV file in JSON format.
- **`/api/get_performance_report`**: Fetches and displays the system performance report.
- **`/download_log`**: Provides a downloadable CSV of detection logs.
- **`/download_alerts`**: Provides a downloadable CSV of alerts.

## Setting Up the Project

1. **Install Dependencies**
   Make sure you have the required packages by installing them from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   Start the Flask application:
   ```bash
   python app.py
   ```

3. **Access the Web Interface**
   Open a browser and go to `http://127.0.0.1:5000/` to upload video files and view real-time detections.

## Usage Guide

1. **Uploading Video Files**:
   - Navigate to the home page and upload a video file in `.mp4` format.
   - Once uploaded, you will be redirected to the dashboard to view the video feed.

2. **Real-Time Monitoring**:
   - The dashboard displays the video with bounding boxes around detected objects.
   - Suspicious activity (e.g., unusual movement of certain objects) triggers an on-screen alert.

3. **Download Logs and Alerts**:
   - The system logs detections and alerts in CSV files, which can be downloaded through the interface.
## Data Visualization:
   Avg Map for  one instance:
   
   ![image](https://github.com/user-attachments/assets/275a6e77-f101-477d-9962-b5c9f08e1b34)

   Avg Fps for one instance:
   
   ![image](https://github.com/user-attachments/assets/85b7c874-8783-4b33-ab21-712a68390df1)


   plot :

   ![image](https://github.com/user-attachments/assets/780285ac-9fb8-41b2-97c5-477fa8ba515c)

   
## Performance Reporting

A performance report is generated in `data/performance_report.txt`, which includes:
- **Average FPS**: Indicates real-time processing capability.
- **Average mAP**: Measures detection accuracy.
- **Strengths and Limitations**: Describes system performance under various conditions, with recommendations for improvement.
  ## generated performace  report:
  ![image](https://github.com/user-attachments/assets/09c6e765-018b-4841-8f08-c8f0dad835db)


## Front End Plus Detection Example:

   Upload Screen:
   ![image](https://github.com/user-attachments/assets/58f0248e-1448-4c7b-8281-96a892b09b13)

   

   DashBoard:
   ![image](https://github.com/user-attachments/assets/f0486b98-1c2b-433e-ac71-da5fca2ab943)


   Suspucious Work Detection:
   ![image](https://github.com/user-attachments/assets/a7706572-90e8-42b5-8444-a7a92143af37)



## Future Enhancements

- **Enhanced Tracking Algorithms**: Implement more advanced tracking techniques to improve detection accuracy.
- **Customizable Alerts**: Enable users to configure alert conditions based on object types or movement patterns.
- **Extended Model Training**: Retrain YOLOv5 on more diverse datasets, including low-light and crowded scenes, to improve robustness.

## Acknowledgments

This project utilizes the YOLOv5 model by Ultralytics for object detection and is implemented with OpenCV, Flask, and Pandas for data handling and visualization.