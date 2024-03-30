import cv2
from ultralytics import YOLO
import sqlite3
import time
from datetime import datetime

people_amounts = []

files = ['video/file1.mp4', 'video/file2.mp4', 'video/file3.mp4', 'video/file4.mp4', 'video/file5.mp4',
         'video/file6.mp4', 'video/file7.mp4', 'video/file8.mp4', 'video/file9.mp4', ]
def average():
    global people_amounts
    return sum(people_amounts) / len(people_amounts) if people_amounts != [] else 0


def get_settings():
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM Settings")
    a = dict(cursor.fetchall())
    connection.commit()
    connection.close()
    return a


def write_data(time, num, cam):
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    cursor.execute("INSERT INTO Data (measure_time, avg_people, camera) VALUES (?, ?, ?)",
                   (time, num, cam))
    connection.commit()
    connection.close()


def draw_data(frame, camera, num_people):
    cv2.rectangle(frame, (0, settings['resy'] - 20), (settings['resx'], settings['resy']), (255, 0, 0), -1)
    string = f'camera: {camera}, number of people: {num_people}. Print Esc to exit program, W or Q to switch cameras'
    cv2.putText(frame,
                string,
                (5, settings['resy'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255),
                2)


settings = get_settings()
if settings['print_data']:
    print(settings)
model = YOLO("runs/heads.pt")

if settings['save_video']: cv2.startWindowThread()
prev_time = time.time()
j = 0
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15.,
                          (settings['resx'], settings['resy'])) if settings['save_video'] else None
while True:
    j %= len(files)
    video_path = files[j]
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        current_time = time.time()
        if current_time - prev_time >= settings['timer']:
            if settings['save_data']:
                current_datetime = datetime.now()
                write_data(current_datetime.strftime("%Y-%m-%d %H:%M:%S"), average(), video_path)
                if settings['print_data']:
                    print("DATA SENT")
            people_amounts = []
            prev_time = current_time
        ret, frame = cap.read()
        if not ret:
            j += 1
            break
        results = model(frame)
        annotated_frame = results[0].plot()

        a = 0
        for result in results:
            a = len(result.boxes.cls)
        people_amounts.append(a)
        if settings['display_video']:

            draw_data(annotated_frame, video_path, a)
            cv2.imshow("OLOvY8 Inference", annotated_frame)
        if settings['save_video']:
            out.write(annotated_frame.astype('uint8'))
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            cap.release()
            if settings['save_video']:
                out.release()
            cv2.destroyAllWindows()
            exit()
        if key & 0xFF == ord('w'):
            j += 1
            break
        if key & 0xFF == ord('q'):
            j -= 1
            break

cap.release()
if settings['save_video']:
    out.release()
cv2.destroyAllWindows()
