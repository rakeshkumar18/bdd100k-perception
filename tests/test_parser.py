from src.ingestion.parser import BDDParser
from src.utils.paths import TRAIN_LABELS


def test_parser_loads_files():

    parser = BDDParser()

    scenes = parser.load_directory(TRAIN_LABELS, max_files=5)

    assert len(scenes) == 5


def test_scene_structure():

    parser = BDDParser()

    scenes = parser.load_directory(TRAIN_LABELS, max_files=1)

    scene = scenes[0]

    assert scene.image_name is not None
    assert isinstance(scene.objects, list)


def test_object_structure():

    parser = BDDParser()

    scenes = parser.load_directory(TRAIN_LABELS, max_files=1)

    scene = scenes[0]

    assert len(scene.objects) > 0

    obj = scene.objects[0]

    assert obj.category is not None
    assert obj.bbox.width > 0
    assert obj.bbox.height > 0
