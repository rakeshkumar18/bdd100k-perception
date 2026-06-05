from ultralytics import YOLO


class YOLOEvaluator:

    def __init__(
        self,
        model_path
    ):
        self.model = YOLO(model_path)

    def evaluate(
        self,
        data_yaml
    ):

        results = self.model.val(
            data=data_yaml,
            split="val"
        )

        return results