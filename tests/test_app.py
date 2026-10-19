import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "server" / "app.py"
MODEL_FILE = tempfile.NamedTemporaryFile(suffix=".task", delete=False)
MODEL_FILE.close()
os.environ["MODEL_PATH"] = MODEL_FILE.name
os.environ["MODEL_SHA256"] = hashlib.sha256(Path(MODEL_FILE.name).read_bytes()).hexdigest()
os.environ["INDEX_PATH"] = str(ROOT / "index.html")


class FakeFaceLandmarker:
    @classmethod
    def create_from_options(cls, _options):
        return cls()

    def detect(self, _image):
        return SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[])


fake_mediapipe = types.ModuleType("mediapipe")
fake_mediapipe.tasks = SimpleNamespace(
    BaseOptions=lambda **kwargs: kwargs,
    vision=SimpleNamespace(
        FaceLandmarker=FakeFaceLandmarker,
        FaceLandmarkerOptions=lambda **kwargs: kwargs,
        RunningMode=SimpleNamespace(IMAGE="IMAGE"),
    ),
)
fake_mediapipe.Image = lambda **kwargs: kwargs
fake_mediapipe.ImageFormat = SimpleNamespace(SRGB="SRGB")
sys.modules["mediapipe"] = fake_mediapipe

spec = importlib.util.spec_from_file_location("yanzhi_app", APP_PATH)
app_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(app_module)


def make_landmarks():
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    coordinates = {
        234: (0.20, 0.50),
        454: (0.80, 0.50),
        10: (0.50, 0.10),
        152: (0.50, 0.90),
        33: (0.32, 0.36),
        133: (0.42, 0.36),
        362: (0.58, 0.36),
        263: (0.68, 0.36),
        61: (0.38, 0.68),
        291: (0.62, 0.68),
        6: (0.50, 0.45),
        2: (0.50, 0.58),
        1: (0.50, 0.50),
    }
    for index, (x, y) in coordinates.items():
        points[index] = SimpleNamespace(x=x, y=y, z=0.0)
    return points


def image_bytes(image_format="JPEG", size=(320, 320), color=(150, 130, 120)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


class DecodeImageTests(unittest.TestCase):
    def test_accepts_supported_static_formats(self):
        for image_format in ("JPEG", "PNG", "WEBP"):
            with self.subTest(image_format=image_format):
                image = app_module.decode_image(image_bytes(image_format=image_format))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (320, 320))

    def test_preserves_exif_orientation_behavior(self):
        buffer = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (320, 240), "white").save(buffer, format="JPEG", exif=exif)
        image = app_module.decode_image(buffer.getvalue())
        self.assertEqual(image.size, (240, 320))

    def test_transparent_png_is_composited_on_white(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (320, 320), (10, 20, 30, 0)).save(buffer, format="PNG")
        image = app_module.decode_image(buffer.getvalue())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_color_key_transparency_is_composited_on_white(self):
        buffer = io.BytesIO()
        Image.new("RGB", (320, 320), (0, 0, 0)).save(
            buffer,
            format="PNG",
            transparency=(0, 0, 0),
        )
        image = app_module.decode_image(buffer.getvalue())
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_rejects_disguised_bmp(self):
        with self.assertRaises(app_module.InvalidImageError):
            app_module.decode_image(image_bytes(image_format="BMP"))

    def test_rejects_animated_webp(self):
        buffer = io.BytesIO()
        frames = [Image.new("RGB", (320, 320), color) for color in ("white", "black")]
        frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=100)
        with self.assertRaises(app_module.UnsupportedImageError):
            app_module.decode_image(buffer.getvalue())

    def test_rejects_excessive_dimension_before_analysis(self):
        with self.assertRaises(app_module.ImageTooLargeError):
            app_module.decode_image(image_bytes(size=(app_module.MAX_IMAGE_DIMENSION + 1, 10)))


class MetricTests(unittest.TestCase):
    def test_face_quality_ignores_background_outside_face_box(self):
        landmarks = make_landmarks()
        first = Image.new("RGB", (400, 400), "black")
        second = Image.new("RGB", (400, 400), "white")
        face_patch = Image.new("RGB", (300, 400), (145, 125, 115))
        first.paste(face_patch, (50, 0))
        second.paste(face_patch, (50, 0))
        first_score, _ = app_module.photo_quality(first, landmarks)
        second_score, _ = app_module.photo_quality(second, landmarks)
        self.assertAlmostEqual(first_score, second_score, places=6)

    def test_photo_composition_penalizes_face_near_frame_edges(self):
        landmarks = make_landmarks()
        central_score, _ = app_module.photo_composition(landmarks)
        edge_landmarks = [SimpleNamespace(x=point.x, y=point.y, z=point.z) for point in landmarks]
        edge_landmarks[234].x = 0.02
        edge_landmarks[454].x = 0.98
        edge_landmarks[10].y = 0.02
        edge_landmarks[152].y = 0.98
        edge_score, details = app_module.photo_composition(edge_landmarks)

        self.assertLess(edge_score, central_score)
        self.assertLess(details["min_margin"], 0.03)

    def test_analysis_image_is_capped_without_upscaling(self):
        large = app_module.resize_for_analysis(Image.new("RGB", (1600, 1200)))
        small = app_module.resize_for_analysis(Image.new("RGB", (640, 480)))

        self.assertEqual(large.size, (1024, 768))
        self.assertEqual(small.size, (640, 480))

    def test_pose_matrix_penalizes_yaw(self):
        identity_score, identity_angles = app_module.pose_from_matrix(np.eye(4))
        radians = np.deg2rad(30)
        yaw_matrix = np.array([
            [np.cos(radians), 0, np.sin(radians), 0],
            [0, 1, 0, 0],
            [-np.sin(radians), 0, np.cos(radians), 0],
            [0, 0, 0, 1],
        ])
        yaw_score, yaw_angles = app_module.pose_from_matrix(yaw_matrix)
        self.assertEqual(identity_score, 1.0)
        self.assertEqual(identity_angles["yaw"], 0.0)
        self.assertLess(yaw_score, identity_score)
        self.assertAlmostEqual(abs(yaw_angles["yaw"]), 30.0, places=1)

    def test_scores_use_full_zero_to_ten_scale(self):
        self.assertEqual(app_module.score10(0.0), 0.0)
        self.assertEqual(app_module.score10(1.0), 10.0)

    def test_presentation_scale_has_explicit_bounds_and_is_monotonic(self):
        self.assertEqual(app_module.presentation_score(0.00), 5.0)
        self.assertEqual(app_module.presentation_score(0.60), 7.7)
        self.assertEqual(app_module.presentation_score(1.00), 9.5)
        self.assertLess(app_module.presentation_score(0.72), app_module.presentation_score(0.84))

    def test_photo_condition_scale_is_more_conservative(self):
        self.assertEqual(app_module.condition_score(0.00), 5.0)
        self.assertEqual(app_module.condition_score(0.67), 6.7)
        self.assertEqual(app_module.condition_score(1.00), 9.5)

    def test_tight_framing_reduces_overall_presentation(self):
        central = app_module.overall_presentation_index(0.70, 0.68, 0.80, 0.67)
        tight = app_module.overall_presentation_index(0.70, 0.68, 0.38, 0.67)

        self.assertGreater(central, tight)
        self.assertLessEqual(app_module.presentation_score(tight), 7.7)

    def test_style_ties_are_reported_as_mixed_and_rank_deterministically(self):
        style = app_module.choose_style({
            "structure": 0.5,
            "symmetry": 0.5,
            "vertical_balance": 0.5,
            "roundness": 0.5,
            "feature_emphasis": 0.5,
            "feature_presence": 0.5,
            "harmony": 0.5,
        })
        self.assertEqual(style["margin"], 0.0)
        self.assertEqual(style["tendency"], "混合倾向")
        self.assertEqual(style["signal"], "两个风格信号接近")
        self.assertNotEqual(style["name"], style["secondary_name"])

    def test_style_returns_primary_secondary_and_tendency(self):
        style = app_module.choose_style({
            "structure": 0.80,
            "symmetry": 0.90,
            "vertical_balance": 0.55,
            "roundness": 0.62,
            "feature_emphasis": 0.86,
            "feature_presence": 0.82,
            "harmony": 0.78,
        })
        self.assertNotEqual(style["name"], style["secondary_name"])
        self.assertIn(style["tendency"], {"混合倾向", "双重倾向", "主风格较明确"})
        self.assertGreaterEqual(style["margin"], 0.0)
        self.assertIn(" × ", style["combo_name"])
        self.assertEqual(len(style["keywords"]), 3)

    def test_report_uses_photo_framing_and_actionable_improvement(self):
        metrics = app_module.measure(make_landmarks(), Image.new("RGB", (320, 320), (150, 130, 120)), np.eye(4))
        metrics["quality_details"] = {"brightness": 60.0, "contrast": 20.0, "sharpness": 1.0}
        metrics["quality"] = 0.55
        report = app_module.build_report(metrics)

        self.assertEqual(
            [section["name"] for section in report["sections"]],
            ["五官比例", "轮廓线条", "照片质感", "上镜稳定性"],
        )
        self.assertEqual(report["improvement"]["title"], "先优化照片条件")
        self.assertIn("只描述这张照片", report["photo_check"]["message"])
        self.assertNotIn("皮相", " ".join(section["name"] for section in report["sections"]))
        self.assertIn(report["photo_check"]["reference"], {"较低", "中等", "较高"})
        self.assertIn("confidence_note", report["photo_check"])
        self.assertIn("calibration", report["method"])
        displayed_scores = [section["score"] for section in report["sections"]]
        displayed_scores.extend(report["overview"].values())
        self.assertTrue(all(5.0 <= score <= 9.5 for score in displayed_scores))

    def test_severe_tight_framing_forces_low_reference(self):
        metrics = app_module.measure(
            make_landmarks(),
            Image.new("RGB", (320, 320), (150, 130, 120)),
            np.eye(4),
        )
        metrics["quality"] = 0.67
        metrics["quality_details"] = {"brightness": 151.0, "contrast": 39.0, "sharpness": 4.1}
        metrics["composition"] = 0.38
        result = app_module.photo_check(metrics)

        self.assertEqual(result["reference"], "较低")
        self.assertIn("构图过近或贴边", result["tags"])
        self.assertIn("清晰度一般", result["tags"])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        self.landmarks = make_landmarks()
        self.original_detect = app_module._landmarker.detect
        self.original_measure = app_module.measure

    def tearDown(self):
        app_module._landmarker.detect = self.original_detect
        app_module.measure = self.original_measure

    def test_oversized_request_body_is_rejected_before_multipart_parsing(self):
        response = self.client.post(
            "/api/analyze",
            content=b"x" * (9 * 1024 * 1024 + 1),
            headers={"content-type": "multipart/form-data; boundary=test-boundary"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("请求体不能超过", response.json()["detail"])

    def test_legitimate_jpeg_keeps_existing_api_contract(self):
        app_module._landmarker.detect = lambda _image: SimpleNamespace(
            face_landmarks=[self.landmarks],
            facial_transformation_matrixes=[np.eye(4)],
        )
        response = self.client.post(
            "/api/analyze",
            files={"photo": ("portrait.jpg", image_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("score", payload)
        self.assertIn("score_label", payload)
        self.assertEqual(set(payload["overview"]), {"feature_score", "photo_score"})
        self.assertIn(payload["photo_check"]["tone"], {"good", "warn"})
        self.assertEqual(len(payload["highlights"]), 2)
        self.assertIn("improvement", payload)
        self.assertIn("secondary_name", payload["style"])
        self.assertIn("tendency", payload["style"])
        self.assertIn("signal", payload["style"])
        self.assertEqual(set(payload["style"]["scores"]), {"清冷感", "甜感", "明艳感", "氛围感"})
        self.assertIn("combo_name", payload["style"])
        self.assertIn("identity", payload["style"])
        self.assertIn("description", payload["style"])
        self.assertIn("share", payload)
        self.assertEqual(len(payload["share"]["formula"]), 3)
        self.assertEqual(len(payload["share"]["keywords"]), 3)
        self.assertIn(payload["photo_check"]["reference"], {"较低", "中等", "较高"})
        self.assertEqual(len(payload["sections"]), 4)
        self.assertEqual(len(payload["advice"]), 3)
        self.assertIn("composition", payload["photo_check"])
        self.assertIn("confidence", payload["photo_check"])
        self.assertIn("calibration", payload["method"])
        self.assertTrue(all("evidence" in section for section in payload["sections"]))
        self.assertTrue(all({"avoid", "action"} <= set(item) for item in payload["advice"]))

    def test_low_photo_quality_is_rejected_before_report(self):
        app_module._landmarker.detect = lambda _image: SimpleNamespace(
            face_landmarks=[self.landmarks],
            facial_transformation_matrixes=[np.eye(4)],
        )
        app_module.measure = lambda *_args: {"quality": 0.19, "pose_stability": 1.0}
        response = self.client.post(
            "/api/analyze",
            files={"photo": ("dark.jpg", image_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("曝光或清晰度不足", response.json()["detail"])

    def test_large_pose_offset_is_rejected_before_report(self):
        app_module._landmarker.detect = lambda _image: SimpleNamespace(
            face_landmarks=[self.landmarks],
            facial_transformation_matrixes=[np.eye(4)],
        )
        app_module.measure = lambda *_args: {"quality": 0.80, "pose_stability": 0.29}
        response = self.client.post(
            "/api/analyze",
            files={"photo": ("side.jpg", image_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("角度偏移较大", response.json()["detail"])

    def test_spoofed_bmp_does_not_reach_model(self):
        calls = []
        app_module._landmarker.detect = lambda image: calls.append(image)
        response = self.client.post(
            "/api/analyze",
            files={"photo": ("fake.png", image_bytes(image_format="BMP"), "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(calls, [])

    def test_multiple_faces_are_rejected(self):
        app_module._landmarker.detect = lambda _image: SimpleNamespace(
            face_landmarks=[self.landmarks, self.landmarks],
            facial_transformation_matrixes=[np.eye(4), np.eye(4)],
        )
        response = self.client.post(
            "/api/analyze",
            files={"photo": ("group.jpg", image_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("多张人脸", response.json()["detail"])


def tearDownModule():
    Path(MODEL_FILE.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
