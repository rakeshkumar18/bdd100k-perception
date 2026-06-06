from pathlib import Path


class YoloYamlGenerator:

    def __init__(self, dataset_root, class_names):

        self.dataset_root = Path(dataset_root)
        self.class_names = class_names

    def generate(self):

        yaml_text = f"""
path: {self.dataset_root}

train: images/100k/train
val: images/100k/val

nc: {len(self.class_names)}

names: {self.class_names}
"""

        yaml_path = self.dataset_root / "dataset.yaml"

        with open(yaml_path, "w") as f:
            f.write(yaml_text.strip())

        return yaml_path
