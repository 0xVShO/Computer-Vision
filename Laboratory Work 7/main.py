import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

class Transport:
    COLORS = {
        "car": (0, 255, 0),
        "truck": (255, 165, 0),
        "bus": (0, 165, 255),
        "motorcycle": (255, 0, 255),
    }
    def __init__(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        label: str,
        conf: float,
    ):
        self.id = track_id
        self.missed = 0
        self.class_scores: dict[str, float] = {label: conf}

        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[4:, 4:] *= 0.01
        self.kf.processNoiseCov[-1, -1] *= 0.01

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.measurementNoiseCov[2:, 2:] *= 10.0

        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 10.0
        self.kf.errorCovPost[4:, 4:] *= 1000.0

        self.kf.statePost = np.zeros((7, 1), dtype=np.float32)
        self.kf.statePost[:4] = self._vectorize(bbox)
        self.bbox = bbox

    @staticmethod
    def _vectorize(bbox: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        u = x1 + w / 2.0
        v = y1 + h / 2.0
        s = w * h
        r = w / h
        return np.array([[u], [v], [s], [r]], dtype=np.float32)

    @staticmethod
    def _devectorize(state: np.ndarray) -> tuple[float, float, float, float]:
        u = float(state[0, 0])
        v = float(state[1, 0])
        s = max(1.0, float(state[2, 0]))
        r = max(0.1, float(state[3, 0]))
        w = np.sqrt(s * r)
        h = s / w
        return (u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0)

    def predict(self) -> tuple[float, float, float, float]:
        if self.kf.statePost[2, 0] + self.kf.statePost[6, 0] <= 0:
            self.kf.statePost[6, 0] = 0.0
        pred = self.kf.predict()
        self.missed += 1
        self.bbox = self._devectorize(pred)
        return self.bbox

    def update(
        self,
        bbox: tuple[float, float, float, float],
        label: str,
        conf: float,
    ) -> None:
        self.missed = 0
        z = self._vectorize(bbox)
        corr = self.kf.correct(z)
        self.bbox = self._devectorize(corr)
        self.class_scores[label] = self.class_scores.get(label, 0.0) + conf

    @property
    def label(self) -> str:
        return max(self.class_scores, key=self.class_scores.get)

    def draw(self, frame: np.ndarray) -> None:
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        lbl = self.label
        color = self.COLORS.get(lbl, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {self.id}: {lbl}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

class ObjectTracker:
    def __init__(
        self,
        video_path: Path,
        model_path: Path,
        target_label: str = "vehicles",
        target_width: int = 800,
        detect_interval: int = 4,
        iou_threshold: float = 0.25,
        conf_threshold: float = 0.45,
        max_missed: int = 12,
        yolo_imgsz: int = 480,
    ):
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.video_path = video_path
        self.video = cv2.VideoCapture(str(video_path))
        if not self.video.isOpened():
            raise RuntimeError("Failed to open video stream")

        self.model = YOLO(str(model_path))
        self.target_label = target_label
        self.target_width = target_width
        self.detect_interval = detect_interval
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.max_missed = max_missed
        self.yolo_imgsz = yolo_imgsz

        ret, init_frame = self.video.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")
        init_frame = self._resize_frame(init_frame)
        self.frame_h, self.frame_w = init_frame.shape[:2]
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.tracks: list[Transport] = []
        self.id_counter = 0
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "fps": [],
            "predict_ms": [],
            "detect_ms": [],
            "compare_ms": [],
            "draw_ms": [],
            "active_counts": [],
            "total_detections": 0,
            "total_matched": 0,
            "removed_missed": 0,
            "removed_oob": 0,
            "counts_by_type": {},
            "timeline": [],
        }

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.target_width:
            return frame
        scale = self.target_width / w
        return cv2.resize(
            frame, (self.target_width, int(h * scale)), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _compute_iou(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _compare(
        self,
        detections: list[tuple[tuple[float, float, float, float], str, float]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, (det_bbox, _, _) in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._compute_iou(track.bbox, det_bbox)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [
            i for i in range(len(self.tracks)) if i not in matched_tracks
        ]
        return matched, unmatched_dets, unmatched_tracks

    def _detect_vehicles(
        self, frame: np.ndarray
    ) -> list[tuple[tuple[float, float, float, float], str, float]]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            imgsz=self.yolo_imgsz,
            classes=[2, 3, 5, 7],
            verbose=False
        )
        detections: list[tuple[tuple[float, float, float, float], str, float]] = []
        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, clss):
                bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                label = res.names.get(int(cls_id), "vehicle")
                detections.append((bbox, label, float(conf)))
        return detections

    def _save_log(
        self, log_path: Path, total_time: float, frames_processed: int
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fps_arr = (
            np.array(self.stats["fps"], dtype=np.float32)
            if self.stats["fps"]
            else np.array([0.0])
        )
        act_arr = (
            np.array(self.stats["active_counts"], dtype=np.float32)
            if self.stats["active_counts"]
            else np.array([0.0])
        )

        def avg_ms(key: str) -> float:
            vals = self.stats[key]
            return float(np.mean(vals)) if vals else 0.0

        def max_ms(key: str) -> float:
            vals = self.stats[key]
            return float(np.max(vals)) if vals else 0.0

        lines = [
            f"=== OBJECT TRACKER PERFORMANCE REPORT ({self.target_label.upper()}) ===",
            f"Video: {self.video_path.name} ({self.frame_w}x{self.frame_h})",
            f"Config: detect_interval={self.detect_interval}, conf={self.conf_threshold}, "
            f"iou={self.iou_threshold}, max_missed={self.max_missed}, imgsz={self.yolo_imgsz}",
            f"Frames processed: {frames_processed} | Total time: {total_time:.2f}s",
            "",
            "--- FPS ---",
            f"Avg: {np.mean(fps_arr):.2f} | Median: {np.median(fps_arr):.2f} | "
            f"P10 (lows): {np.percentile(fps_arr, 10):.2f} | Min: {np.min(fps_arr):.2f} | Max: {np.max(fps_arr):.2f}",
            "",
            "--- TIMING BREAKDOWN (ms per call) ---",
            f"Kalman predict  : avg={avg_ms('predict_ms'):.2f} ms | max={max_ms('predict_ms'):.2f} ms",
            f"YOLO detect     : avg={avg_ms('detect_ms'):.2f} ms | max={max_ms('detect_ms'):.2f} ms",
            f"SORT compare    : avg={avg_ms('compare_ms'):.2f} ms | max={max_ms('compare_ms'):.2f} ms",
            f"Draw + write    : avg={avg_ms('draw_ms'):.2f} ms | max={max_ms('draw_ms'):.2f} ms",
            "",
            "--- DETECTION & TRACKER STATS ---",
            f"Active trackers : avg={np.mean(act_arr):.2f} | max={int(np.max(act_arr))}",
            f"Total created   : {self.id_counter}",
            f"YOLO detections : {self.stats['total_detections']} -> matched: {self.stats['total_matched']}, "
            f"new: {self.id_counter}",
            f"Removed reasons : missed={self.stats['removed_missed']}, oob={self.stats['removed_oob']}",
            f"Counts by type  : {self.stats['counts_by_type']}",
            "",
            "--- TIMELINE (every 30 frames) ---",
            "frame | fps   | active | predict_ms | detect_ms | compare_ms | detections",
        ]
        lines.extend(self.stats["timeline"])

        log_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[УСПІХ] Лог збережено у {log_path}")

    def process(
        self,
        log_path: Path | None = Path("logs/tracker_log_sort.txt"),
        output_video: Path | None = Path("logs/output_sort.mp4"),
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
                for track in self.tracks:
                    track.predict()
                predict_ms = (time.perf_counter() - t0) * 1000.0
                self.stats["predict_ms"].append(predict_ms)

                detect_ms = 0.0
                compare_ms = 0.0
                det_count = 0

                if frame_idx == 1 or frame_idx % self.detect_interval == 0:
                    t1 = time.perf_counter()
                    detections = self._detect_vehicles(frame)
                    detect_ms = (time.perf_counter() - t1) * 1000.0
                    self.stats["detect_ms"].append(detect_ms)
                    det_count = len(detections)
                    self.stats["total_detections"] += det_count

                    t2 = time.perf_counter()
                    matched, unmatched_dets, _ = self._compare(detections)

                    for t_idx, d_idx in matched:
                        bbox, label, conf = detections[d_idx]
                        self.tracks[t_idx].update(bbox, label, conf)

                    for d_idx in unmatched_dets:
                        bbox, label, conf = detections[d_idx]
                        self.id_counter += 1
                        self.tracks.append(Transport(self.id_counter, bbox, label, conf))

                    compare_ms = (time.perf_counter() - t2) * 1000.0
                    self.stats["compare_ms"].append(compare_ms)
                    self.stats["total_matched"] += len(matched)

                kept_tracks: list[Transport] = []
                for t in self.tracks:
                    if t.missed > self.max_missed:
                        self.stats["removed_missed"] += 1
                        self.stats["counts_by_type"][t.label] = (
                            self.stats["counts_by_type"].get(t.label, 0) + 1
                        )
                    elif (
                        t.bbox[2] <= 0
                        or t.bbox[3] <= 0
                        or t.bbox[0] >= self.frame_w
                        or t.bbox[1] >= self.frame_h
                    ):
                        self.stats["removed_oob"] += 1
                        self.stats["counts_by_type"][t.label] = (
                            self.stats["counts_by_type"].get(t.label, 0) + 1
                        )
                    else:
                        kept_tracks.append(t)
                self.tracks = kept_tracks

                t3 = time.perf_counter()
                now = time.perf_counter()
                fps = 1.0 / max(1e-6, now - prev_time)
                prev_time = now

                for track in self.tracks:
                    track.draw(frame)

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | Active: {len(self.tracks)} | Total: {self.id_counter}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                if writer is not None:
                    writer.write(frame)

                cv2.imshow("ObjectTracker", frame)
                key = cv2.waitKey(1) & 0xFF
                draw_ms = (time.perf_counter() - t3) * 1000.0

                self.stats["draw_ms"].append(draw_ms)
                self.stats["fps"].append(fps)
                self.stats["active_counts"].append(len(self.tracks))

                if frame_idx % 30 == 0:
                    self.stats["timeline"].append(
                        f"{frame_idx:5d} | {fps:5.1f} | {len(self.tracks):6d} | "
                        f"{predict_ms:10.2f} | {detect_ms:9.2f} | {compare_ms:10.2f} | {det_count}"
                    )

                if key == 27:
                    break
        finally:
            for t in self.tracks:
                self.stats["counts_by_type"][t.label] = (
                    self.stats["counts_by_type"].get(t.label, 0) + 1
                )
            total_time = time.perf_counter() - start_total
            self.video.release()
            if writer is not None:
                writer.release()
                print(f"[УСПІХ] Відео збережено у {output_video}")
            cv2.destroyAllWindows()
            if log_path is not None and frame_idx > 0:
                self._save_log(log_path, total_time, frame_idx)


if __name__ == "__main__":
    video_path = Path("sources/test.mp4")
    model_path = Path("models/yolov8n.pt")

    tracker = ObjectTracker(video_path=video_path, model_path=model_path)
    tracker.process(
        log_path=Path("logs/tracker_log_sort.txt"),
        output_video=Path("logs/output_sort.mp4"),
    )