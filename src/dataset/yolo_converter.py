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

    def convert_df(self, df: pd.DataFrame, output_dir: str):

        output_dir = Path(output_dir)
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
                    row["x1"], row["y1"], row["x2"], row["y2"]
                )

                lines.append(
                    f"{class_id} " f"{x_c:.6f} " f"{y_c:.6f} " f"{w:.6f} " f"{h:.6f}"
                )

            label_file = output_dir / f"{Path(image_name).stem}.txt"

            with open(label_file, "w") as f:
                f.write("\n".join(lines))

    def convert_csv(self, csv_path, output_dir):

        df = pd.read_csv(csv_path)

        self.convert_df(df=df, output_dir=output_dir)
