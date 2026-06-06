from src.dataset.dataset_validator import DatasetValidator


def main():

    validator = DatasetValidator()

    print("=" * 60)
    print("BDD100K DATASET VALIDATION")
    print("=" * 60)

    missing = validator.validate_structure()

    if missing:

        print("\nMissing paths:")

        for path in missing:
            print(path)

        return

    print("\n✓ Directory structure valid")

    for split in ["train", "val", "test"]:

        print("\n" + "-" * 40)
        print(split.upper())
        print("-" * 40)

        pair_results = validator.validate_pairs(split)

        print(f"Missing labels: " f"{len(pair_results['missing_labels'])}")

        print(f"Missing images: " f"{len(pair_results['missing_images'])}")

        corrupted = validator.validate_images(split)

        print(f"Corrupted images: " f"{len(corrupted)}")

        empty_labels = validator.count_empty_labels(split)

        print(f"Empty labels: " f"{len(empty_labels)}")

        invalid_format = validator.validate_label_format(split)

        print(f"Invalid label rows: " f"{len(invalid_format)}")

        invalid_boxes = validator.validate_bbox_ranges(split)

        print(f"Invalid bbox ranges: " f"{len(invalid_boxes)}")

    print("\n✓ Validation Complete")


if __name__ == "__main__":
    main()
