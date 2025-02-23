from roboflow import Roboflow
from ultralytics import YOLO 

rf = Roboflow(api_key="key")
project = rf.workspace("clauirtontrack").project("tennis-detection-mqipj")
version = project.version(1)
dataset = version.download("yolov5")

!yolo task=detect mode=train model=yolov5l6u.pt data={dataset.location}/data.yaml epochs=100 batch=8 imgsz=640


model = YOLO('/home/user/proj_track/models/yolov5l6u.pt')

result = model.track('input_videos/input_video.mp4',conf=0.2, save=True)
