from ultralytics import YOLO
import cv2
import supervision as sv

model = YOLO('./runs/detect/train2/weights/best.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

while cap.isOpened():
  ret, img = cap.read()
  if not ret:
    break
  result = model.predict(img)

  detections = sv.Detections.from_ultralytics(result[0])
  labels = [model.model.names[class_id] for class_id in detections.class_id]

  annotated_frame = bounding_box_annotator.annotate(scene=img.copy(), detections=detections)
  annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

  cv2.imshow("webcam", annotated_frame)
  if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()