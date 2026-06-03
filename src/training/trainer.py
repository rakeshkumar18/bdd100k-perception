from ultralytics import YOLO


class YOLOTrainer:

    def __init__(
        self,
        model_name="yolov8n.pt"
    ):
        self.model = YOLO(model_name)

    def train(
        self,
        data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
    ):

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device="mps",
            workers=4,
            cache=True,
            project="outputs/training",
            name="yolov8n_bdd100k",
            exist_ok=True,
            verbose=True,
        )

        return results
from ultralytics import YOLO


class YOLOTrainer:

    def __init__(
        self,
        model_name="yolov8n.pt"
    ):
        self.model = YOLO(model_name)

    def train(
        self,
        data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
    ):

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device="mps",
            workers=4,
            cache=True,
            project="outputs/training",
            name="yolov8n_bdd100k",
            exist_ok=True,
            verbose=True,
        )

        return results