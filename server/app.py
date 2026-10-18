from __future__ import annotations

import io
import math
import os
import threading
import warnings
from pathlib import Path

import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "models" / "face_landmarker.task"))
if index_path_env := os.environ.get("INDEX_PATH"):
    INDEX_PATH = Path(index_path_env)
else:
    index_candidates = (BASE_DIR / "static" / "index.html", BASE_DIR.parent / "index.html")
    INDEX_PATH = next((path for path in index_candidates if path.is_file()), index_candidates[0])
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_DIMENSION = 6_500
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(title="面部风格分析服务", version="1.4.0")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

if not MODEL_PATH.is_file():
    raise RuntimeError(f"离线模型不存在: {MODEL_PATH}")
if not INDEX_PATH.is_file():
    raise RuntimeError(f"前端页面不存在: {INDEX_PATH}")

_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=2,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        output_facial_transformation_matrixes=True,
    )
)
_model_lock = threading.Lock()
_request_slots = threading.BoundedSemaphore(2)


class UnsupportedImageError(ValueError):
    pass


class InvalidImageError(ValueError):
    pass


class ImageTooLargeError(ValueError):
    pass


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def near(value: float, target: float, tolerance: float) -> float:
    return clamp(1.0 - abs(value - target) / tolerance)


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def decode_image(content: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content), formats=ALLOWED_FORMATS) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise UnsupportedImageError("图片真实格式不受支持")
                width, height = probe.size
                if width <= 0 or height <= 0:
                    raise InvalidImageError("图片尺寸异常")
                if max(width, height) > MAX_IMAGE_DIMENSION or width * height > MAX_IMAGE_PIXELS:
                    raise ImageTooLargeError("图片像素尺寸过大")
                if getattr(probe, "n_frames", 1) != 1:
                    raise UnsupportedImageError("不支持动态图片")
                probe.verify()

            with Image.open(io.BytesIO(content), formats=ALLOWED_FORMATS) as opened:
                opened.draft("RGB", (1800, 1800))
                opened.load()
                image = ImageOps.exif_transpose(opened)
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    image = image.convert("RGB")
                return image.copy()
    except UnsupportedImageError:
        raise
    except ImageTooLargeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageTooLargeError("图片像素尺寸过大") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise InvalidImageError("无法识别这张图片") from None


def face_bounds(landmarks) -> tuple[float, float, float, float] | None:
    xs = [point.x for point in landmarks if math.isfinite(point.x)]
    ys = [point.y for point in landmarks if math.isfinite(point.y)]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def face_region(image: Image.Image, landmarks) -> Image.Image:
    bounds = face_bounds(landmarks)
    if bounds is None:
        return image

    width, height = image.size
    left, top, right, bottom = (value * scale for value, scale in zip(bounds, (width, height, width, height)))
    pad_x = (right - left) * 0.08
    pad_y = (bottom - top) * 0.08
    box = (
        max(0, int(left - pad_x)),
        max(0, int(top - pad_y)),
        min(width, int(right + pad_x)),
        min(height, int(bottom + pad_y)),
    )
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        return image
    return image.crop(box)


def photo_composition(landmarks) -> tuple[float, dict[str, float]]:
    """Estimate whether the face has enough framing room for a stable observation."""
    bounds = face_bounds(landmarks)
    if bounds is None:
        return 0.0, {"face_area": 0.0, "min_margin": 0.0, "center_offset": 1.0}

    left, top, right, bottom = bounds
    face_width = max(0.0, right - left)
    face_height = max(0.0, bottom - top)
    face_area = face_width * face_height
    min_margin = min(left, top, 1.0 - right, 1.0 - bottom)
    center_offset = math.hypot((left + right) / 2.0 - 0.5, (top + bottom) / 2.0 - 0.5)

    edge_score = clamp(min_margin / 0.06)
    coverage_score = clamp(1.0 - abs(face_area - 0.28) / 0.30)
    center_score = clamp(1.0 - center_offset / 0.42)
    composition = clamp(edge_score * 0.55 + coverage_score * 0.25 + center_score * 0.20)
    return composition, {
        "face_area": round(face_area, 4),
        "min_margin": round(min_margin, 4),
        "center_offset": round(center_offset, 4),
    }


def photo_quality(image: Image.Image, landmarks) -> tuple[float, dict[str, float]]:
    region = face_region(image, landmarks)
    gray = ImageOps.grayscale(region).resize((160, 160), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    softened = np.asarray(gray.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32)
    gradient_x = float(np.abs(np.diff(softened, axis=1)).mean())
    gradient_y = float(np.abs(np.diff(softened, axis=0)).mean())
    sharpness = (gradient_x + gradient_y) / 2

    exposure_score = clamp(1.0 - abs(brightness - 145.0) / 125.0)
    contrast_score = clamp(contrast / 52.0)
    sharpness_score = clamp((sharpness - 1.5) / 9.0)
    quality = exposure_score * 0.40 + contrast_score * 0.25 + sharpness_score * 0.35
    return quality, {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
    }


def pose_from_matrix(matrix) -> tuple[float, dict[str, float]]:
    try:
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError
        rotation = transform[:3, :3]
        u, _, vh = np.linalg.svd(rotation)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        horizontal = math.hypot(rotation[0, 0], rotation[1, 0])
        if horizontal > 1e-6:
            pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
            yaw = math.degrees(math.atan2(-rotation[2, 0], horizontal))
            roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rotation[1, 2], rotation[1, 1]))
            yaw = math.degrees(math.atan2(-rotation[2, 0], horizontal))
            roll = 0.0
        angles = {"pitch": pitch, "yaw": yaw, "roll": roll}
        stability = float(np.mean([
            clamp(1.0 - abs(pitch) / 25.0),
            clamp(1.0 - abs(yaw) / 30.0),
            clamp(1.0 - abs(roll) / 18.0),
        ]))
        return stability, {name: round(value, 1) for name, value in angles.items()}
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return 0.0, {}


def measure(landmarks, image: Image.Image, pose_matrix=None) -> dict[str, float | dict[str, float]]:
    p = lambda index: landmarks[index]
    face_width = distance(p(234), p(454))
    face_height = distance(p(10), p(152))
    if face_width <= 0 or face_height <= 0:
        raise ValueError("面部关键点距离异常")

    left_eye = distance(p(33), p(133))
    right_eye = distance(p(362), p(263))
    eye_gap = distance(p(133), p(362))
    mouth_width = distance(p(61), p(291))
    nose_length = distance(p(6), p(2))
    eye_level = abs((p(33).y + p(133).y) / 2 - (p(362).y + p(263).y) / 2)
    upper_face = distance(p(10), p(1))
    lower_face = distance(p(1), p(152))

    face_ratio = face_height / face_width
    eye_gap_ratio = eye_gap / face_width
    mouth_ratio = mouth_width / face_width
    upper_lower_ratio = upper_face / lower_face
    eye_symmetry = 1.0 - abs(left_eye - right_eye) / max(left_eye, right_eye)
    outline_balance = distance(p(234), p(1)) / distance(p(1), p(454))
    harmony = float(np.mean([
        near(face_ratio, 1.35, 0.75),
        near(eye_gap_ratio, 0.24, 0.18),
        near(mouth_ratio, 0.34, 0.22),
        near(upper_lower_ratio, 0.95, 0.70),
    ]))
    symmetry = clamp(float(np.mean([
        eye_symmetry,
        1.0 - eye_level / face_width,
        near(outline_balance, 1.0, 0.70),
    ])))
    structure = clamp(near(face_ratio, 1.35, 0.80))
    feature_presence = clamp(float(np.mean([
        left_eye / face_width,
        mouth_ratio,
        nose_length / face_height,
    ])) * 3.2)
    roundness = clamp(1.0 - structure * 0.65 + near(face_ratio, 1.15, 0.75) * 0.35)
    vertical_balance = clamp((p(1).y - p(10).y) / face_height)
    feature_emphasis = clamp(feature_presence * 0.55 + symmetry * 0.25 + (1.0 - roundness) * 0.20)
    quality, quality_details = photo_quality(image, landmarks)
    pose_stability, pose_angles = pose_from_matrix(pose_matrix)
    if not pose_angles:
        eye_roll = clamp(1.0 - eye_level / max(face_width * 0.10, 1e-6))
        eye_mid_x = (p(33).x + p(133).x + p(362).x + p(263).x) / 4
        yaw = abs(p(1).x - eye_mid_x) / face_width
        pose_stability = float(np.mean([eye_roll, clamp(1.0 - yaw / 0.12)]))

    composition, composition_details = photo_composition(landmarks)
    feature_score = clamp(
        harmony * 0.36
        + symmetry * 0.24
        + structure * 0.15
        + feature_presence * 0.25
    )
    photo_condition = clamp(
        pose_stability * 0.35
        + quality * 0.45
        + composition * 0.20
    )
    raw_index = clamp(feature_score * 0.82 + photo_condition * 0.18)
    score = presentation_score(raw_index)
    return {
        "harmony": harmony,
        "symmetry": symmetry,
        "structure": structure,
        "feature_presence": feature_presence,
        "roundness": roundness,
        "vertical_balance": vertical_balance,
        "feature_emphasis": feature_emphasis,
        "quality": quality,
        "pose_stability": pose_stability,
        "composition": composition,
        "feature_score": feature_score,
        "photo_condition": photo_condition,
        "face_ratio": face_ratio,
        "eye_gap_ratio": eye_gap_ratio,
        "mouth_ratio": mouth_ratio,
        "upper_lower_ratio": upper_lower_ratio,
        "eye_symmetry": eye_symmetry,
        "outline_balance": outline_balance,
        "raw_index": raw_index,
        "score": score,
        "quality_details": quality_details,
        "composition_details": composition_details,
        "pose_angles": pose_angles,
    }


def score10(value: float) -> float:
    return round(clamp(value) * 10.0, 1)


def presentation_score(value: float) -> float:
    """Map a 0-1 rule index to the product's 5.0-9.5 presentation scale."""
    return round(5.0 + 4.5 * clamp((value - 0.60) / 0.40), 1)


def choose_style(metrics: dict) -> dict:
    candidates = [
        {
            "name": "清冷感",
            "value": metrics["structure"] * 0.45 + metrics["symmetry"] * 0.25 + (1 - metrics["vertical_balance"]) * 0.30,
            "description": "面部线条和视觉留白偏利落、克制，干净、有边界感的造型更容易突出优势。",
            "tags": ["低饱和妆容", "利落线条", "简洁配色"],
        },
        {
            "name": "甜感",
            "value": metrics["roundness"] * 0.45 + metrics["vertical_balance"] * 0.35 + metrics["symmetry"] * 0.20,
            "description": "面部圆润度和亲和感更明显，柔和、轻盈的造型更容易放大优势。",
            "tags": ["自然眉形", "柔和腮红", "轻盈发型"],
        },
        {
            "name": "明艳感",
            "value": metrics["feature_emphasis"] * 0.45 + metrics["feature_presence"] * 0.35 + metrics["symmetry"] * 0.20,
            "description": "五官存在感相对突出，明确但不过度堆叠的妆发重点更容易放大优势。",
            "tags": ["强调眼唇", "明快色彩", "轮廓感发型"],
        },
        {
            "name": "氛围感",
            "value": metrics["harmony"] * 0.45 + metrics["symmetry"] * 0.30 + metrics["roundness"] * 0.25,
            "description": "整体比例比较平衡，风格空间较大，光线、表情和搭配会明显改变呈现。",
            "tags": ["自然光线", "层次穿搭", "松弛感"],
        },
    ]
    ranked = sorted(candidates, key=lambda item: item["value"], reverse=True)
    primary, secondary = ranked[:2]
    margin = primary["value"] - secondary["value"]
    tendency = "混合倾向" if margin < 0.04 else "双重倾向" if margin < 0.10 else "主风格较明确"
    signal = "风格信号较明确" if margin >= 0.10 else "两个风格信号接近" if margin < 0.04 else "存在明显的双重倾向"
    pair = frozenset((primary["name"], secondary["name"]))
    combo_copy = {
        frozenset(("清冷感", "明艳感")): ("高辨识度与克制感并存", "五官有存在感，越懂得留白，越容易显出高级和记忆点。"),
        frozenset(("清冷感", "甜感")): ("清透边界里的柔和感", "利落线条里带着亲和感，轻盈而不甜腻的造型更适合你。"),
        frozenset(("清冷感", "氛围感")): ("安静、耐看、有距离感", "你的优势不在堆叠元素，而在光线、留白和整体色调的一致。"),
        frozenset(("甜感", "明艳感")): ("亲和感里带着视觉重点", "柔和底色配一个明确重点，比一味减淡或一味加强都更出彩。"),
        frozenset(("甜感", "氛围感")): ("柔和松弛的氛围型", "轻盈材质、自然表情与同色系搭配，最容易放大你的亲和力。"),
        frozenset(("明艳感", "氛围感")): ("有重点的氛围型", "你适合先定一个视觉中心，再让色彩与材质围绕它形成完整气氛。"),
    }
    identity, share_line = combo_copy[pair]
    combo_name = f"{primary['name'].removesuffix('感')} × {secondary['name'].removesuffix('感')}"
    keywords = list(dict.fromkeys([
        primary["tags"][0],
        secondary["tags"][1],
        "单点强调" if metrics["feature_presence"] >= 0.78 else "轻盈留白",
    ]))[:3]
    return {
        "name": primary["name"],
        "description": primary["description"],
        "tags": keywords,
        "secondary_name": secondary["name"],
        "tendency": tendency,
        "signal": signal,
        "margin": round(margin, 3),
        "scores": {item["name"]: round(item["value"], 3) for item in ranked},
        "combo_name": combo_name,
        "identity": identity,
        "share_line": share_line,
        "keywords": keywords,
    }


def photo_check(metrics: dict) -> dict:
    details = metrics["quality_details"]
    angles = metrics["pose_angles"]
    tags = []
    warnings_found = []

    brightness = details["brightness"]
    if brightness < 85:
        tags.append("曝光偏暗")
        warnings_found.append("补充正面自然光")
    elif brightness > 200:
        tags.append("曝光偏亮")
        warnings_found.append("避免面部高光过曝")
    else:
        tags.append("曝光正常")

    if details["contrast"] < 28:
        tags.append("对比较弱")
        warnings_found.append("减少雾化或强磨皮")
    else:
        tags.append("对比清楚")

    if details["sharpness"] < 2.4:
        tags.append("清晰度一般")
        warnings_found.append("保持镜头稳定并对焦眼睛")
    else:
        tags.append("清晰度可用")

    if metrics["pose_stability"] < 0.72:
        tags.append("角度有偏移")
        warnings_found.append("尽量使用接近正面的照片")
    else:
        tags.append("正脸稳定")

    composition = metrics["composition"]
    if composition < 0.55:
        tags.append("构图过近或贴边")
        warnings_found.append("让面部四周保留更多空间")
    elif composition < 0.75:
        tags.append("构图较近")
        warnings_found.append("尽量不要把面部贴到画面边缘")
    else:
        tags.append("构图空间稳定")

    confidence = clamp(
        metrics["quality"] * 0.45
        + metrics["pose_stability"] * 0.25
        + composition * 0.30
    )
    if confidence >= 0.78:
        reference = "较高"
        confidence_note = "光线、清晰度、朝向与构图共同满足较稳定的观察条件。"
    elif confidence >= 0.52:
        reference = "中等"
        confidence_note = "可以观察整体关系，但照片条件会影响部分细节的判断。"
    else:
        reference = "较低"
        confidence_note = "建议先改善照片条件，再把这次结果用于妆发或搭配比较。"

    if warnings_found or reference == "较低":
        return {
            "status": f"分析可用 · 参考度{reference}",
            "tone": "warn",
            "tags": tags,
            "message": "；".join(dict.fromkeys(warnings_found)) + ("，" if warnings_found else "") + f"{confidence_note}结果只描述这张照片，复测会更稳定。",
            "angles": angles,
            "reference": reference,
            "composition": round(composition, 2),
            "confidence": round(confidence, 2),
            "confidence_note": confidence_note,
        }
    return {
        "status": f"分析参考度{reference}",
        "tone": "good",
        "tags": tags,
        "message": f"{confidence_note}结果只描述这张照片。",
        "angles": angles,
        "reference": reference,
        "composition": round(composition, 2),
        "confidence": round(confidence, 2),
        "confidence_note": confidence_note,
    }


def build_report(metrics: dict) -> dict:
    style = choose_style(metrics)
    if metrics["score"] >= 8.6:
        score_label = "这张照片的风格信号很清晰"
        summary = f"在这张照片里，你呈现出明显的{style['combo_name']}倾向。{style['share_line']}"
    elif metrics["score"] >= 7.6:
        score_label = "这张照片里的优势比较容易被看见"
        summary = f"在这张照片里，你更接近{style['combo_name']}。{style['share_line']}"
    else:
        score_label = "这张照片已经呈现出个人特点"
        summary = f"当前照片更接近{style['combo_name']}，调整光线或造型后，风格信号可能会更稳定。"

    sections = [
        {
            "name": "五官比例",
            "score": presentation_score(metrics["harmony"]),
            "observation": "眉眼、鼻唇之间的视觉重量比较均衡，第一眼不会被单一区域打断。" if metrics["harmony"] >= 0.72 else "五官大小关系有个人辨识度，镜头距离会明显改变整体观感。",
            "evidence": "基于眼距、嘴宽与面部纵横关系的相对比例观察，不使用绝对尺寸。",
            "suggestion": "保留自然眉眼关系，只选眼妆或唇色中的一个作为视觉中心。",
        },
        {
            "name": "轮廓线条",
            "score": presentation_score(metrics["structure"] * 0.65 + metrics["symmetry"] * 0.35),
            "observation": "面部纵向延伸感更明显，利落线条和适度侧分容易放大轮廓优势。" if metrics["face_ratio"] >= 1.34 else "面部外轮廓更显柔和，轻盈层次与顶部蓬松感更容易拉开比例。",
            "evidence": "观察面部纵横关系、中心线与左右外轮廓的照片呈现；拍摄角度变化会影响结果。",
            "suggestion": "让发型在颧骨到下颌附近保留层次，避免两侧同时完全贴脸。",
        },
        {
            "name": "照片质感",
            "score": score10(metrics["quality"]),
            "observation": "当前画面的曝光、对比与清晰度共同决定了面部细节在照片里的呈现。",
            "evidence": "这一项只衡量照片成像，不等同于真实肤质、皮肤状态或医学判断。",
            "suggestion": "优先使用窗边正面自然光，减少强磨皮与顶光，再判断妆面质感。",
        },
        {
            "name": "上镜稳定性",
            "score": score10(float(np.mean([metrics["harmony"], metrics["symmetry"], metrics["quality"], metrics["pose_stability"]]))),
            "observation": "面部朝向稳定，当前结果对左右关系的参考价值较高。" if metrics["pose_stability"] >= 0.72 else "面部有轻微偏转，左右关系会受到角度影响，但仍可观察整体风格。",
            "evidence": "综合三轴面部姿态、左右眼位与画面清晰度，判断这张照片的分析可用性。",
            "suggestion": "镜头略高于眼睛，肩膀放松，面部只做很小幅度的侧转会更自然。",
        },
    ]

    makeup_base = {
        "清冷感": "轻薄底妆配自然灰棕眉，眼唇都控制在低饱和范围，保留面部留白。",
        "甜感": "柔和眉形、轻盈腮红和水润唇色，重点是弱化妆容边界而不是堆叠可爱元素。",
        "明艳感": "先统一底妆，再从眼妆和唇妆中只选一个作为重点，五官会更有记忆点。",
        "氛围感": "保持底妆通透，用同色温的眼、颊、唇建立完整氛围。",
    }[style["name"]]
    if metrics["eye_gap_ratio"] >= 0.27:
        eye_action = "眼线起点与眉头可稍向内收，但边缘保持柔和。"
    elif metrics["eye_gap_ratio"] <= 0.21:
        eye_action = "眼尾与眉尾可轻微向外延伸，给眉眼增加呼吸感。"
    else:
        eye_action = "保持自然眼距，用睫毛根部与眉形清晰度提升精神感。"
    makeup = (
        f"{makeup_base}{eye_action}",
        "眼、颊、唇同时使用高饱和或强边界，让多个区域互相抢重点。",
        f"先完成轻薄底妆，再执行这一条：{eye_action}",
    )
    hair_text = (
        "优先考虑有线条感的中长层次或轻微侧分，在颧骨到下颌处保留少量弧度。"
        if metrics["face_ratio"] >= 1.34
        else "可尝试轻盈层次、自然弧度和适度头顶蓬松，增加纵向延伸感。"
    )
    hair = (
        hair_text,
        "厚重齐刘海与两侧完全贴脸同时出现。" if metrics["face_ratio"] < 1.34 else "顶部过高、两侧完全收紧的极端纵向拉伸。",
        "同一光线下分别拍一张露出下颌线、再拍一张保留两侧碎发，直接对比哪张更显轻盈。",
    )
    outfit = {
        "清冷感": ("黑白灰、低饱和蓝绿和简洁剪裁更容易保持整体统一。", "多种高饱和颜色同时抢重点。", "用黑白灰基础款搭配一件低饱和蓝绿色单品。"),
        "甜感": ("柔和浅色、圆润配饰和轻盈材质更容易延续亲和感。", "大面积硬挺深色与尖锐配饰叠加。", "从柔和浅色上装和一件圆润配饰开始。"),
        "明艳感": ("选择一件有颜色或设计重点的单品，其他部分保持干净。", "全身每个区域都有强设计点。", "确定一个主色单品，其余使用中性色收住。"),
        "氛围感": ("从同一色系和有层次的材质开始，再用小面积配饰增加记忆点。", "色温、材质和图案同时产生冲突。", "先做同色系搭配，再加入一件有纹理的单品。"),
    }[style["name"]]

    ranked_sections = sorted(sections, key=lambda item: item["score"], reverse=True)
    highlights = [
        {"title": item["name"], "text": item["observation"]}
        for item in ranked_sections[:2]
    ]
    advice = [
        {"title": "化妆建议", "label": "单点强调", "text": makeup[0], "avoid": makeup[1], "action": makeup[2]},
        {"title": "发型建议", "label": "修饰轮廓", "text": hair[0], "avoid": hair[1], "action": hair[2]},
        {"title": "日常搭配", "label": "延续气质", "text": outfit[0], "avoid": outfit[1], "action": outfit[2]},
    ]
    reference = photo_check(metrics)
    if reference["tone"] == "warn":
        improvement = {
            "title": "先优化照片条件",
            "text": f"{reference['message']} 建议用相近构图重新拍摄后，再比较妆发与穿搭效果。",
        }
    elif metrics["eye_gap_ratio"] >= 0.27:
        improvement = {"title": "先收束眉眼重点", "text": eye_action}
    elif metrics["eye_gap_ratio"] <= 0.21:
        improvement = {"title": "先给眉眼增加延伸", "text": eye_action}
    elif metrics["face_ratio"] >= 1.34:
        improvement = {"title": "先试轻侧分层次", "text": hair_text}
    else:
        improvement = {"title": "先试轻盈顶部层次", "text": hair_text}

    return {
        "score": round(metrics["score"], 1),
        "score_label": score_label,
        "summary": summary,
        "overview": {
            "feature_score": presentation_score(metrics["feature_score"]),
            "photo_score": score10(metrics["photo_condition"]),
        },
        "style": {
            "name": style["name"],
            "description": style["description"],
            "tags": style["tags"],
            "secondary_name": style["secondary_name"],
            "tendency": style["tendency"],
            "signal": style["signal"],
            "combo_name": style["combo_name"],
            "identity": style["identity"],
            "share_line": style["share_line"],
            "scores": style["scores"],
        },
        "photo_check": reference,
        "highlights": highlights,
        "improvement": improvement,
        "sections": sections,
        "advice": advice,
        "share": {
            "headline": f"我测出来是「{style['combo_name']}」",
            "identity": style["identity"],
            "line": style["share_line"],
            "keywords": style["keywords"],
            "reference": f"本次照片参考度{reference['reference']}",
            "formula": [
                {"title": "妆容公式", "text": makeup[2]},
                {"title": "发型公式", "text": hair[2]},
                {"title": "搭配公式", "text": outfit[2]},
            ],
        },
        "method": {
            "model": "MediaPipe Face Landmarker (offline)",
            "metrics": "面部关键点、相对比例、左右平衡、三轴姿态、构图边界，以及面部区域的曝光、对比度和清晰度",
            "notice": "分数是由规则指标映射到 5.0–9.5 的实验性照片呈现量表，只描述当前照片，不是人群百分位、医学诊断或客观颜值结论。",
            "calibration": "当前权重来自可解释的规则设计，尚未用足量、脱敏、取得授权的人群数据做统计校准；结果适合做风格探索，不适合做人与人之间的排名。",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH.name, "offline": True, "version": app.version}


@app.post("/api/analyze")
def analyze(photo: UploadFile = File(...)) -> dict:
    content_type = (photo.content_type or "").lower().split(";", 1)[0].strip()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "仅支持 JPG、PNG 或 WebP 图片")
    if not _request_slots.acquire(blocking=False):
        raise HTTPException(429, "当前分析人数较多，请稍后重试")
    try:
        content = photo.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "图片不能超过 8 MB")
        if not content:
            raise HTTPException(400, "图片内容为空")
        try:
            image = decode_image(content)
        except UnsupportedImageError as error:
            raise HTTPException(415, str(error)) from None
        except ImageTooLargeError:
            raise HTTPException(413, "图片像素尺寸过大，请压缩到 2000 万像素以内") from None
        except InvalidImageError as error:
            raise HTTPException(400, str(error)) from None

        if min(image.size) < 240:
            raise HTTPException(400, "图片分辨率过低，短边至少需要 240 像素")
        image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
        image_array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
        with _model_lock:
            result = _landmarker.detect(mp_image)
        if not result.face_landmarks:
            raise HTTPException(422, "未检测到清晰正脸，请更换无遮挡、光线充足的照片")
        if len(result.face_landmarks) != 1:
            raise HTTPException(422, "检测到多张人脸，请上传只有一人的照片")

        pose_matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
        try:
            metrics = measure(result.face_landmarks[0], image, pose_matrix)
        except (ValueError, ZeroDivisionError):
            raise HTTPException(422, "面部关键点质量不足，请更换更清晰的正脸照片") from None
        if metrics["quality"] < 0.20:
            raise HTTPException(422, "照片曝光或清晰度不足，请在自然光下重新拍摄")
        if metrics["pose_stability"] < 0.30:
            raise HTTPException(422, "面部角度偏移较大，请上传更接近正面的照片")
        if metrics["composition"] < 0.25:
            raise HTTPException(422, "面部裁切或构图过近，请让脸部四周保留更多空间")
        return build_report(metrics)
    finally:
        _request_slots.release()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_PATH)
