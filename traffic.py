from ultralytics import YOLO
import cv2

# Load YOLOv8 model (use 'yolov8n' for the small model, or 'yolov8s', 'yolov8m', etc. for others)
model = YOLO("yolov8n.pt")

video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize vehicle counter
vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Perform vehicle detection (e.g., on each frame)
    results = model(frame)

    # Filter the results to detect vehicles (class ID 2 corresponds to 'car' in COCO dataset)
    vehicles = [result for result in results[0].boxes.cls if result in [
        2, 3, 5, 7]]  # 2: Car, 3: Motorcycle, 5: Bus, 7: Truck

    # Count the number of detected vehicles
    vehicle_count = len(vehicles)

    # Display the frame with bounding boxes
    for result in results[0].boxes:
        if result.cls in [2, 3, 5, 7]:
            if result.cls == 2:
                veh = "Car"
            elif result.cls == 3:
                veh = "Motorcycle"
            elif result.cls == 5:
                veh = "Bus"
            elif result.cls == 7:
                veh = "Truck"
            else:
                pass

            x1, y1, x2, y2 = result.xyxy[0].tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(
                x2), int(y2)), (255, 0, 0), 2)  # Blue color
            cv2.putText(frame, veh, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Traffic Video", frame)

    # Optional: Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print the current vehicle count per frame (optional)
    print(f"Detected vehicles: {vehicle_count}")

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
