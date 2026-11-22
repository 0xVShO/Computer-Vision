import numpy as np
import cv2

class MotionDetector:
    def __init__(
        self,
        frame_shape: tuple[int, int],
        history: int = 300,
        var_threshold: int = 40,
        min_area: int = 450,
        warmup_frames: int = 15,
    ):
        """Ініціалізація початкових параметрів детектора руху"""
        self.frame_h, self.frame_w = frame_shape
        self.min_area = min_area
        self.max_area = int(self.frame_h * self.frame_w * 0.25)
        self.warmup_frames = warmup_frames
        self.frame_count = 0

        # Блок ініціалізації фонового віднімача (detectShadows=True позначає тіні як 127, а рух — як 255)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )

        # Блок створення морфологічних ядер для очищення маски
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        self.last_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)

    def detect(
        self,
        frame: cv2.typing.MatLike,
        active_bboxes: list[tuple[int, int, int, int]],
        pad_ratio: float = 0.15,
    ) -> list[tuple[int, int, int, int]]:
        """Метод пошуку нових рухомих об'єктів у кадрі, повертає список рамок (x, y, w, h)"""
        self.frame_count += 1
        fg_mask = self.bg_subtractor.apply(frame)

        # Даємо MOG2 кілька перших кадрів, щоб накопичити модель статичного фону
        if self.frame_count <= self.warmup_frames:
            return []

        # Залишаємо лише впевнений рух (255), відсікаючи тіні (127)
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Прибираємо дрібний шум та склеюємо частини одного об'єкта в єдину пляму
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.close_kernel)
        self.last_mask = binary.copy()

        # Блок маскування областей, які вже захоплені активними трекерами
        for x, y, w, h in active_bboxes:
            pad_x = int(w * pad_ratio)
            pad_y = int(h * pad_ratio)
            x1 = max(0, int(x) - pad_x)
            y1 = max(0, int(y) - pad_y)
            x2 = min(self.frame_w, int(x + w) + pad_x)
            y2 = min(self.frame_h, int(y + h) + pad_y)
            binary[y1:y2, x1:x2] = 0

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Блок фільтрації контурів за площею та мінімальними розмірами
        candidates: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue

            candidates.append((x, y, w, h))

        return candidates