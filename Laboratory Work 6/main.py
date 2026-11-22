import time
from pathlib import Path
import cv2
import joblib
import numpy as np
from motion_grid import MotionDetector
from train_model import extract_features


class ObjectTracker:
    def __init__(
        self,
        video: Path,
        model: Path | None = None,
        target_label: str | None = None,
        target_width: int = 800,
        iou_threshold: float = 0.25,
        ios_threshold: float = 0.45,
        confidence_threshold: float = 0.72,
        max_static_frames: int = 20,
        detect_interval: int = 3,
        mask_pad_ratio: float = 0.35,
    ):
        """Ініціалізація початкових параметрів трекера об'єктів"""
        self.video_path = self._validate_path(video, "відео")
        self.video = cv2.VideoCapture(str(self.video_path))
        if not self.video.isOpened():
            raise RuntimeError("Не вдалося відкрити відеопотік!")

        # Блок завантаження класифікатора та назви цільового класу
        self.model = None
        self.target_label = target_label or "object"
        if model is not None:
            loaded = joblib.load(self._validate_path(model, "моделі"))
            if isinstance(loaded, dict):
                self.model = loaded.get("model")
                classes = loaded.get("classes")
                if target_label is None and classes and len(classes) > 1:
                    self.target_label = str(classes[1])
            else:
                self.model = loaded

            if self.model is not None and hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1

        ret, init_frame = self.video.read()
        if not ret:
            raise RuntimeError("Не вдалося прочитати перший кадр відео!")

        self.target_width = target_width
        init_frame = self._resize_frame(init_frame)
        self.frame_h, self.frame_w = init_frame.shape[:2]

        self.detector = MotionDetector(frame_shape=(self.frame_h, self.frame_w))
        self.active_trackers: list[dict] = []
        self.id_counter = 0

        self.iou_threshold = iou_threshold
        self.ios_threshold = ios_threshold
        self.confidence_threshold = confidence_threshold
        self.max_static_frames = max_static_frames
        self.detect_interval = detect_interval
        self.mask_pad_ratio = mask_pad_ratio

        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "fps": [],
            "track_ms": [],
            "detect_ms": [],
            "classify_ms": [],
            "draw_ms": [],
            "active_counts": [],
            "total_candidates": 0,
            "total_confirmed": 0,
            "skipped_duplicates": 0,
            "removed_lost": 0,
            "removed_oob": 0,
            "removed_static": 0,
            "removed_dedup": 0,
            "timeline": [],
        }

    @staticmethod
    def _validate_path(path: Path | None, file_label: str) -> Path:
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Файл {file_label} не знайдено за шляхом {path}! Перевірте чи існує файл."
            )
        return path

    def _resize_frame(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        h, w = frame.shape[:2]
        if w <= self.target_width:
            return frame
        scale = self.target_width / w
        return cv2.resize(
            frame, (self.target_width, int(h * scale)), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _create_tracker():
        if hasattr(cv2, "TrackerKCF"):
            return cv2.TrackerKCF.create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        return cv2.TrackerCSRT.create()

    @staticmethod
    def _compute_overlap(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> tuple[float, float]:
        ax1, ay1, aw, ah = box_a
        bx1, by1, bw, bh = box_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh
        union_area = area_a + area_b - inter_area
        min_area = min(area_a, area_b)

        iou = inter_area / union_area if union_area > 0 else 0.0
        ios = inter_area / min_area if min_area > 0 else 0.0
        return iou, ios

    def _is_overlapping(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> bool:
        iou, ios = self._compute_overlap(box_a, box_b)
        return iou >= self.iou_threshold or ios >= self.ios_threshold

    def _classify_candidates(
        self,
        frame: cv2.typing.MatLike,
        candidates: list[tuple[int, int, int, int]],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        """Класифікує знайдені області руху за допомогою моделі RandomForest"""
        if not candidates:
            return []

        if self.model is None:
            return [(self.target_label, bbox) for bbox in candidates]

        features_batch: list[np.ndarray] = []
        valid_boxes: list[tuple[int, int, int, int]] = []

        for x, y, w, h in candidates:
            patch = frame[y : y + h, x : x + w]
            if patch.size == 0:
                continue
            features_batch.append(extract_features(patch))
            valid_boxes.append((x, y, w, h))

        if not features_batch:
            return []

        probs = self.model.predict_proba(np.array(features_batch, dtype=np.float32))
        target_probs = probs[:, 1]

        confirmed: list[tuple[str, tuple[int, int, int, int]]] = []
        for bbox, prob in zip(valid_boxes, target_probs):
            if prob >= self.confidence_threshold:
                confirmed.append((f"{self.target_label} {prob:.2f}", bbox))

        return confirmed

    def _is_duplicate(self, new_bbox: tuple[int, int, int, int]) -> bool:
        return any(
            self._is_overlapping(new_bbox, obj["bbox"])
            for obj in self.active_trackers
        )

    def _add_tracker(
        self,
        frame: cv2.typing.MatLike,
        bbox: tuple[int, int, int, int],
        label: str,
    ) -> None:
        tracker = self._create_tracker()
        tracker.init(frame, bbox)
        self.id_counter += 1
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        self.active_trackers.append(
            {
                "id": self.id_counter,
                "tracker": tracker,
                "label": label,
                "bbox": bbox,
                "anchor": (cx, cy),
                "static_frames": 0,
            }
        )

    def _deduplicate_trackers(self) -> None:
        kept: list[dict] = []
        for obj in sorted(self.active_trackers, key=lambda t: t["id"]):
            if not any(self._is_overlapping(obj["bbox"], k["bbox"]) for k in kept):
                kept.append(obj)
            else:
                self.stats["removed_dedup"] += 1
        self.active_trackers = kept

    def _update_trackers(self, frame: cv2.typing.MatLike) -> None:
        updated: list[dict] = []
        for obj in self.active_trackers:
            ok, new_bbox = obj["tracker"].update(frame)
            if not ok:
                self.stats["removed_lost"] += 1
                continue

            x, y, w, h = [int(v) for v in new_bbox]
            if x + w <= 0 or y + h <= 0 or x >= self.frame_w or y >= self.frame_h:
                self.stats["removed_oob"] += 1
                continue

            cx, cy = x + w / 2, y + h / 2
            ax, ay = obj["anchor"]
            dist = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5

            if dist < 6.0:
                obj["static_frames"] += 1
            else:
                obj["anchor"] = (cx, cy)
                obj["static_frames"] = 0

            if obj["static_frames"] > self.max_static_frames:
                self.stats["removed_static"] += 1
                continue

            obj["bbox"] = (x, y, w, h)
            updated.append(obj)

        self.active_trackers = updated
        self._deduplicate_trackers()

    def _draw_debug(
        self,
        frame: cv2.typing.MatLike,
        candidates: list[tuple[int, int, int, int]],
        fps: float,
    ) -> None:
        for x, y, w, h in candidates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        for obj in self.active_trackers:
            x, y, w, h = obj["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {obj['id']}: {obj['label']}",
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Mode: {self.target_label} | FPS: {fps:.1f} | Trackers: {len(self.active_trackers)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    def _save_log(self, log_path: Path, total_time: float, frames_processed: int) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fps_arr = np.array(self.stats["fps"], dtype=np.float32) if self.stats["fps"] else np.array([0.0])
        act_arr = np.array(self.stats["active_counts"], dtype=np.float32) if self.stats["active_counts"] else np.array([0.0])

        def avg_ms(key: str) -> float:
            vals = self.stats[key]
            return float(np.mean(vals)) if vals else 0.0

        def max_ms(key: str) -> float:
            vals = self.stats[key]
            return float(np.max(vals)) if vals else 0.0

        lines = [
            f"=== OBJECT TRACKER PERFORMANCE REPORT ({self.target_label.upper()}) ===",
            f"Video: {self.video_path.name} ({self.frame_w}x{self.frame_h})",
            f"Config: detect_interval={self.detect_interval}, conf={self.confidence_threshold}, "
            f"iou={self.iou_threshold}, ios={self.ios_threshold}, pad={self.mask_pad_ratio}",
            f"Frames processed: {frames_processed} | Total time: {total_time:.2f}s",
            "",
            "--- FPS ---",
            f"Avg: {np.mean(fps_arr):.2f} | Median: {np.median(fps_arr):.2f} | "
            f"P10 (lows): {np.percentile(fps_arr, 10):.2f} | Min: {np.min(fps_arr):.2f} | Max: {np.max(fps_arr):.2f}",
            "",
            "--- TIMING BREAKDOWN (ms per call) ---",
            f"Trackers update : avg={avg_ms('track_ms'):.2f} ms | max={max_ms('track_ms'):.2f} ms",
            f"MOG2 detect     : avg={avg_ms('detect_ms'):.2f} ms | max={max_ms('detect_ms'):.2f} ms",
            f"RF classify     : avg={avg_ms('classify_ms'):.2f} ms | max={max_ms('classify_ms'):.2f} ms",
            f"Draw + write    : avg={avg_ms('draw_ms'):.2f} ms | max={max_ms('draw_ms'):.2f} ms",
            "",
            "--- DETECTION & TRACKER STATS ---",
            f"Active trackers : avg={np.mean(act_arr):.2f} | max={int(np.max(act_arr))}",
            f"Total created   : {self.id_counter}",
            f"MOG2 candidates : {self.stats['total_candidates']} -> RF confirmed: {self.stats['total_confirmed']} "
            f"(skipped duplicates: {self.stats['skipped_duplicates']})",
            f"Removed reasons : lost_kcf={self.stats['removed_lost']}, oob={self.stats['removed_oob']}, "
            f"static={self.stats['removed_static']}, dedup={self.stats['removed_dedup']}",
            "",
            "--- TIMELINE (every 30 frames) ---",
            "frame | fps   | active | track_ms | detect_ms | classify_ms | candidates",
        ]
        lines.extend(self.stats["timeline"])

        log_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[УСПІХ] Лог збережено у {log_path}")

    def process(
        self,
        debug: bool = True,
        log_path: Path | None = Path("logs/tracker_log.txt"),
        output_video: Path | None = Path("logs/output.mp4"),
    ) -> None:
        self._reset_stats()
        start_total = time.perf_counter()
        prev_time = start_total
        frame_idx = 0

        writer = None
        if output_video is not None:
            output_video.parent.mkdir(parents=True, exist_ok=True)
            src_fps = self.video.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_video), fourcc, src_fps, (self.frame_w, self.frame_h)
            )

        try:
            while self.video.isOpened():
                ret, frame = self.video.read()
                if not ret:
                    break

                frame_idx += 1
                frame = self._resize_frame(frame)

                t0 = time.perf_counter()
                self._update_trackers(frame)
                track_ms = (time.perf_counter() - t0) * 1000.0
                self.stats["track_ms"].append(track_ms)

                candidates: list[tuple[int, int, int, int]] = []
                detect_ms = 0.0
                classify_ms = 0.0

                if frame_idx % self.detect_interval == 0:
                    t1 = time.perf_counter()
                    active_bboxes = [obj["bbox"] for obj in self.active_trackers]
                    candidates = self.detector.detect(
                        frame, active_bboxes, pad_ratio=self.mask_pad_ratio
                    )
                    detect_ms = (time.perf_counter() - t1) * 1000.0
                    self.stats["detect_ms"].append(detect_ms)
                    self.stats["total_candidates"] += len(candidates)

                    t2 = time.perf_counter()
                    confirmed = self._classify_candidates(frame, candidates)
                    classify_ms = (time.perf_counter() - t2) * 1000.0
                    self.stats["classify_ms"].append(classify_ms)
                    self.stats["total_confirmed"] += len(confirmed)

                    for label, bbox in confirmed:
                        if not self._is_duplicate(bbox):
                            self._add_tracker(frame, bbox, label)
                        else:
                            self.stats["skipped_duplicates"] += 1

                t3 = time.perf_counter()
                now = time.perf_counter()
                fps = 1.0 / max(1e-6, now - prev_time)
                prev_time = now

                if debug:
                    self._draw_debug(frame, candidates, fps)

                if writer is not None:
                    writer.write(frame)

                cv2.imshow("ObjectTracker", frame)
                key = cv2.waitKey(1) & 0xFF
                draw_ms = (time.perf_counter() - t3) * 1000.0

                self.stats["draw_ms"].append(draw_ms)
                self.stats["fps"].append(fps)
                self.stats["active_counts"].append(len(self.active_trackers))

                if frame_idx % 30 == 0:
                    self.stats["timeline"].append(
                        f"{frame_idx:5d} | {fps:5.1f} | {len(self.active_trackers):6d} | "
                        f"{track_ms:8.2f} | {detect_ms:9.2f} | {classify_ms:11.2f} | {len(candidates)}"
                    )

                if key == 27:
                    break
        finally:
            total_time = time.perf_counter() - start_total
            self.video.release()
            if writer is not None:
                writer.release()
                print(f"[УСПІХ] Відео збережено у {output_video}")
            cv2.destroyAllWindows()
            if log_path is not None and frame_idx > 0:
                self._save_log(log_path, total_time, frame_idx)


if __name__ == "__main__":
    print("=== ВИБІР РЕЖИМУ ДЕТЕКЦІЇ ===")
    print("1 - Машини (models/rf_cars.joblib)")
    print("2 - Люди (models/rf_people.joblib)")
    choice = input("Введіть 1 або 2: ").strip()

    if choice == "2":
        mode_name = "people"
        box_label = "person"
        model_path = Path("models/rf_people.joblib")
    else:
        mode_name = "cars"
        box_label = "car"
        model_path = Path("models/rf_cars.joblib")

    video_path = Path("sources/test_1.mp4")

    tracker = ObjectTracker(
        video=video_path,
        model=model_path,
        target_label=box_label,
        target_width=800,
        confidence_threshold=0.72,
    )
    tracker.process(
        debug=True,
        log_path=Path(f"logs/tracker_log_{mode_name}.txt"),
        output_video=Path(f"logs/output_{mode_name}.mp4"),
    )