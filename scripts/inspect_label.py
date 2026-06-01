import json
from pathlib import Path
from pprint import pprint

sample_file = next(
    Path("../data/bdd100k/labels/100k/train").glob("*.json")
)

with open(sample_file) as f:
    data = json.load(f)

print("\nTOP LEVEL")
print(data.keys())

print("\nFRAME COUNT")
print(len(data["frames"]))

print("\nFIRST FRAME KEYS")
print(data["frames"][0].keys())

print("\nFIRST FRAME")
pprint(data["frames"][0])