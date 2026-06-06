def extract_detections(result):

    detections = []

    for box in result.boxes:

        cls_id = int(box.cls.item())

        detections.append(
            {
                "class": result.names[cls_id],
                "confidence": round(float(box.conf.item()), 3),
            }
        )

    return detections
