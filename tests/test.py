from pathlib import Path

sample = df.iloc[0]

print(sample["image_name"])

label_file = TRAIN_LABELS / f"{sample['image_name']}.json"

print(label_file)
print(label_file.exists())
