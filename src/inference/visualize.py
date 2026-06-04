import cv2


def get_annotated_image(result):

    return result.plot()


def save_prediction_image(
    result,
    output_path
):

    annotated = result.plot()

    cv2.imwrite(
        output_path,
        annotated
    )