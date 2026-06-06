from ultralytics import YOLO


class YOLOPredictor:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path, conf=0.25):

        results = self.model.predict(source=image_path, conf=conf, verbose=False)

        return results[0]
