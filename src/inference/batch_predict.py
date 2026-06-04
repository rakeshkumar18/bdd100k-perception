from pathlib import Path


def get_images(folder):

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    images = []

    for ext in image_extensions:

        images.extend(
            Path(folder).glob(ext)
        )

    return sorted(images)