from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# import opencv to display our annotated images
import cv2
# import supervision to help visualize our predictions
import supervision as sv

import json

# Load configuration from JSON file
from pathlib import Path

config_path = Path(__file__).parent / "roboflow_config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# create a simple box annotator to use in our custom sink
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the text labels for each prediction
    labels = [f"{p['class']} ({p['confidence']:.2f})" for p in predictions["predictions"]]
    # load our predictions into the Supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = box_annotator.annotate(
        scene=video_frame.image.copy(), detections=detections)
    image = label_annotator.annotate(
        scene=image, detections=detections, labels=labels)
    # display the annotated image
    cv2.imshow("Predictions", image)
    if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(1) == ord('q'):
        pipeline.terminate()
        cv2.destroyAllWindows()

pipeline = InferencePipeline.init(
    model_id=config['ROBOFLOW_MODEL'],
    api_key=config['ROBOFLOW_API_KEY'],
    video_reference=0,
    on_prediction=my_custom_sink,
)

pipeline.start()
pipeline.join()