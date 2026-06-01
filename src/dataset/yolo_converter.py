from pathlib import Path
import pandas as pd


class YOLOConverter:

    def __init__(self, class_map):

        self.class_map = class_map

    def convert_bbox(self, x1, y1, x2, y2, img_w=1280, img_h=720):

        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h

        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        return x_center, y_center, w, h

    def convert_csv(
        self,
        csv_path: str,
        output_label_dir: str
    ):

        df = pd.read_csv(csv_path)
        output_dir = Path(output_label_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        grouped = df.groupby("image_name")

        for image_name, group in grouped:

            lines = []

            for _, row in group.iterrows():

                cls = row["category"]

                if cls not in self.class_map:
                    continue

                class_id = self.class_map[cls]

                x_c, y_c, w, h = self.convert_bbox(
                    row["x1"],
                    row["y1"],
                    row["x2"],
                    row["y2"]
                )

                lines.append(
                    f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                )

            label_file = output_dir / f"{image_name.replace('.jpg','')}.txt"

            with open(label_file, "w") as f:
                f.write("\n".join(lines))