"""Read plates from spaced observations of a tracked vehicle."""
from collections import Counter
from pathlib import Path
import re

import cv2
import onnxruntime as ort


class PlateReader:
    def __init__(self):
        from fast_alpr.default_ocr import DefaultOCR
        from open_image_models import create_detector

        models = Path(__file__).resolve().parent / "plate_models"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.detector = create_detector(
            models / "yolo-v9-t-384-license-plates-end2end.onnx",
            backend="yolo_v9", class_labels=["license_plate"], conf_thresh=.45,
            providers=["CPUExecutionProvider"], sess_options=options,
        )
        self.ocr = DefaultOCR(
            hub_ocr_model=None, device="cpu", sess_options=options,
            model_path=models / "cct_xs_v2_global.onnx",
            config_path=models / "cct_xs_v2_global_plate_config.yaml",
        )

    def read(self, vehicle):
        if vehicle.shape[0] < 60 or vehicle.shape[1] < 95:
            return None
        candidates = []
        for detection in self.detector.predict(vehicle):
            box = detection.bounding_box
            plate = vehicle[max(0, box.y1):box.y2, max(0, box.x1):box.x2]
            if not plate.size or plate.shape[1] < 40 or plate.shape[0] < 14:
                continue
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 35:
                continue
            reading = self.ocr.predict(plate)
            if reading is None:
                continue
            text = re.sub(r"[^A-Z0-9]", "", reading.text.upper())
            confidence = reading.confidence
            if not isinstance(confidence, (int, float)):
                confidence = sum(confidence) / len(confidence) if len(confidence) else 0
            if 5 <= len(text) <= 10 and confidence >= .75:
                candidates.append(text)
        if len(candidates) == 1:
            return candidates[0]
        return None


def confirmed_plate(reads):
    if len(reads) < 3:
        return None
    text, votes = Counter(reads).most_common(1)[0]
    if votes >= 3 and votes / len(reads) >= .75:
        return text
    return None
