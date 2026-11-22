import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
from train_model import extract_features


@dataclass
class Track:
    track_id: int
    kcf: cv2.Tracker
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    history: List[Tuple[float, float]] = field(default_factory=list)
    missed_frames: int = 0


class ObjectTracker:
    def __init__(
        self,
        video: Path,
        model: Path,
        target_label: str = "car",
        target_width: int = 800,
        confidence_threshold: float = 0.72,
        detect_interval: int = 5,
        iou_threshold: float = 0.25,
        ios_threshold: float = 0.45,
        pad_ratio: float = 0.35,
    ):
        self.video_path = Path(video)
        self.model_path = Path(model)
        self.target_label = target_label
        self.target_width = target_width
        self.conf_thresh = confidence_threshold
        self.detect_interval = detect_interval
        self.iou_thresh = iou_threshold
        self.ios_thresh = ios_threshold
        self.pad_ratio = pad_ratio

        # Загрузка Random Forest (с поддержкой формата словаря из train_model.py)
        loaded = joblib.load(self.model_path)
        self.clf = loaded.get("model") if isinstance(loaded, dict) else loaded
        if self.clf is not None and hasattr(self.clf, "n_jobs"):
            self.clf.n_jobs = 1

        # Инициализация вычитателя фона MOG2
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=True
        )

    def _create_kcf(self) -> cv2.Tracker:
        if hasattr(cv2, "TrackerKCF"):
            return cv2.TrackerKCF.create()
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        return cv2.legacy.TrackerKCF_create()

    def _get_mog2_candidates(
        self, frame: np.ndarray, frame_w: int, frame_h: int
    ) -> List[Tuple[int, int, int, int]]:
        fg_mask = self.bg_subtractor.apply(frame)
        # Убираем тени (в MOG2 тени имеют значение 127, объекты — 255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        min_area = (frame_w * frame_h) * 0.0015  # Отсекаем мелкий шум

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            # Добавляем отступ (pad) вокруг контура для лучшего захвата контекста
            pad_w = int(w * self.pad_ratio)
            pad_h = int(h * self.pad_ratio)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame_w, x + w + pad_w)
            y2 = min(frame_h, y + h + pad_h)

            if (x2 - x1) > 15 and (y2 - y1) > 15:
                candidates.append((x1, y1, x2 - x1, y2 - y1))

        return candidates

    def _classify_candidates(
        self, frame: np.ndarray, candidates: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if not candidates or self.clf is None:
            return []

        features_list = []
        valid_boxes = []

        for box in candidates:
            x, y, w, h = box
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            features_list.append(extract_features(crop))
            valid_boxes.append(box)

        if not features_list:
            return []

        X = np.array(features_list, dtype=np.float32)
        probs = self.clf.predict_proba(X)

        target_idx = 1
        if hasattr(self.clf, "classes_"):
            classes = list(self.clf.classes_)
            if self.target_label in classes:
                target_idx = classes.index(self.target_label)
            elif 1 in classes:
                target_idx = classes.index(1)

        confirmed = []
        for box, prob in zip(valid_boxes, probs):
            score = float(prob[target_idx])
            if score >= self.conf_thresh:
                confirmed.append((box, score))

        # Сортируем по убыванию уверенности
        confirmed.sort(key=lambda item: item[1], reverse=True)
        return confirmed

    @staticmethod
    def _compute_iou_and_ios(
        boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """Возвращает IoU (Intersection over Union) и IoS (Intersection over Smaller area)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0, 0.0

        areaA = boxA[2] * boxA[3]
        areaB = boxB[2] * boxB[3]
        union_area = float(areaA + areaB - inter_area)
        smaller_area = float(min(areaA, areaB))

        iou = inter_area / union_area if union_area > 0 else 0.0
        ios = inter_area / smaller_area if smaller_area > 0 else 0.0
        return iou, ios

    @staticmethod
    def _is_out_of_bounds(
        box: Tuple[int, int, int, int], frame_w: int, frame_h: int
    ) -> bool:
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        return cx < 0 or cy < 0 or cx > frame_w or cy > frame_h or w <= 5 or h <= 5

    @staticmethod
    def _is_static(history: List[Tuple[float, float]], window: int = 35) -> bool:
        """Проверяет, не застрял ли трекер на неподвижном фоне."""
        if len(history) < window:
            return False
        recent = np.array(history[-window:])
        displacement = np.linalg.norm(recent[-1] - recent[0])
        spread = np.max(np.ptp(recent, axis=0))
        return displacement < 4.0 and spread < 6.0

    def process(
        self,
        debug: bool = False,
        log_path: Optional[Path] = None,
        output_video: Optional[Path] = None,
    ):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {self.video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

        scale = self.target_width / float(orig_w)
        frame_w = self.target_width
        frame_h = int(round(orig_h * scale))

        writer = None
        if output_video:
            output_video = Path(output_video)
            output_video.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_video), fourcc, fps_in, (frame_w, frame_h)
            )

        tracks: List[Track] = []
        next_track_id = 1
        frame_idx = 0

        # Метрики производительности и статистика
        frame_fps_list: List[float] = []
        track_times_ms: List[float] = []
        detect_times_ms: List[float] = []
        classify_times_ms: List[float] = []
        draw_times_ms: List[float] = []
        active_counts: List[int] = []

        total_created = 0
        total_candidates = 0
        total_confirmed = 0
        skipped_duplicates = 0
        removed_reasons = {"lost_kcf": 0, "oob": 0, "static": 0, "dedup": 0}
        timeline_rows: List[str] = []

        total_start_t = time.perf_counter()

        try:
            while True:
                ret, raw_frame = cap.read()
                if not ret:
                    break

                frame_start_t = time.perf_counter()
                frame_idx += 1
                frame = cv2.resize(raw_frame, (frame_w, frame_h))

                # 1. ОБНОВЛЕНИЕ СУЩЕСТВУЮЩИХ KCF-ТРЕКЕРОВ
                t0 = time.perf_counter()
                surviving_tracks: List[Track] = []
                for trk in tracks:
                    ok, new_box = trk.kcf.update(frame)
                    if not ok:
                        removed_reasons["lost_kcf"] += 1
                        continue

                    bx, by, bw, bh = map(int, new_box)
                    if self._is_out_of_bounds((bx, by, bw, bh), frame_w, frame_h):
                        removed_reasons["oob"] += 1
                        continue

                    trk.bbox = (bx, by, bw, bh)
                    trk.history.append((bx + bw / 2.0, by + bh / 2.0))

                    if self._is_static(trk.history):
                        removed_reasons["static"] += 1
                        continue

                    surviving_tracks.append(trk)

                tracks = surviving_tracks
                track_ms = (time.perf_counter() - t0) * 1000.0
                track_times_ms.append(track_ms)

                # Дедупликация перекрывающихся активных трекеров (оставляем более старый ID)
                deduped_tracks: List[Track] = []
                for trk in tracks:
                    is_dup = False
                    for kept in deduped_tracks:
                        iou, ios = self._compute_iou_and_ios(trk.bbox, kept.bbox)
                        if iou > self.iou_thresh or ios > self.ios_thresh:
                            is_dup = True
                            break
                    if is_dup:
                        removed_reasons["dedup"] += 1
                    else:
                        deduped_tracks.append(trk)
                tracks = deduped_tracks

                # 2. ДЕТЕКЦИЯ MOG2 + КЛАССИФИКАЦИЯ RANDOM FOREST (каждые detect_interval кадров)
                detect_ms = 0.0
                classify_ms = 0.0
                cand_count = 0

                if frame_idx % self.detect_interval == 0:
                    t_det = time.perf_counter()
                    candidates = self._get_mog2_candidates(frame, frame_w, frame_h)
                    detect_ms = (time.perf_counter() - t_det) * 1000.0
                    detect_times_ms.append(detect_ms)

                    cand_count = len(candidates)
                    total_candidates += cand_count

                    if cand_count > 0:
                        t_cls = time.perf_counter()
                        confirmed_boxes = self._classify_candidates(frame, candidates)
                        classify_ms = (time.perf_counter() - t_cls) * 1000.0
                        classify_times_ms.append(classify_ms)
                        total_confirmed += len(confirmed_boxes)

                        # Сопоставление подтвержденных боксов с активными треками
                        for box, score in confirmed_boxes:
                            matched_trk = None
                            best_iou = 0.0
                            for trk in tracks:
                                iou, ios = self._compute_iou_and_ios(box, trk.bbox)
                                if iou > self.iou_thresh or ios > self.ios_thresh:
                                    if iou >= best_iou:
                                        best_iou = iou
                                        matched_trk = trk

                            if matched_trk is not None:
                                # Обновляем (реинициализируем) существующий трекер свежими координатами
                                skipped_duplicates += 1
                                new_kcf = self._create_kcf()
                                new_kcf.init(frame, box)
                                matched_trk.kcf = new_kcf
                                matched_trk.bbox = box
                            else:
                                # Создаем новый трекер
                                new_kcf = self._create_kcf()
                                new_kcf.init(frame, box)
                                cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
                                tracks.append(
                                    Track(
                                        track_id=next_track_id,
                                        kcf=new_kcf,
                                        bbox=box,
                                        history=[(cx, cy)],
                                    )
                                )
                                next_track_id += 1
                                total_created += 1
                else:
                    # Обновляем модель фона на промежуточных кадрах
                    t_det = time.perf_counter()
                    self.bg_subtractor.apply(frame, learningRate=0.005)
                    detect_ms = (time.perf_counter() - t_det) * 1000.0
                    detect_times_ms.append(detect_ms)

                # 3. ОТРИСОВКА И ЗАПИСЬ ВИДЕО
                t_draw = time.perf_counter()
                if writer is not None or debug:
                    vis_frame = frame.copy()
                    for trk in tracks:
                        x, y, w, h = trk.bbox
                        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{self.target_label} #{trk.track_id}"
                        cv2.putText(
                            vis_frame,
                            label,
                            (x, max(15, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                        )
                    if writer is not None:
                        writer.write(vis_frame)
                    if debug:
                        cv2.imshow("AI Object Tracker", vis_frame)
                        if cv2.waitKey(1) & 0xFF == 27:  # ESC для досрочного выхода
                            break

                draw_ms = (time.perf_counter() - t_draw) * 1000.0
                draw_times_ms.append(draw_ms)

                # Расчет мгновенного FPS кадра
                frame_elapsed = time.perf_counter() - frame_start_t
                inst_fps = 1.0 / frame_elapsed if frame_elapsed > 0 else 0.0
                frame_fps_list.append(inst_fps)
                active_counts.append(len(tracks))

                # Логирование в TIMELINE каждые 30 кадров
                if frame_idx % 30 == 0:
                    timeline_rows.append(
                        f"{frame_idx:5d} | {inst_fps:5.1f} | {len(tracks):6d} | "
                        f"{track_ms:8.2f} | {detect_ms:9.2f} | {classify_ms:11.2f} | {cand_count}"
                    )
        finally:
            total_time = time.perf_counter() - total_start_t
            cap.release()
            if writer is not None:
                writer.release()
            if debug:
                cv2.destroyAllWindows()

        # Формирование финального отчета
        report = self._build_report(
            frame_w=frame_w,
            frame_h=frame_h,
            frames_processed=frame_idx,
            total_time=total_time,
            frame_fps_list=frame_fps_list,
            track_times_ms=track_times_ms,
            detect_times_ms=detect_times_ms,
            classify_times_ms=classify_times_ms,
            draw_times_ms=draw_times_ms,
            active_counts=active_counts,
            total_created=total_created,
            total_candidates=total_candidates,
            total_confirmed=total_confirmed,
            skipped_duplicates=skipped_duplicates,
            removed_reasons=removed_reasons,
            timeline_rows=timeline_rows,
        )
        print(report)
        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(report, encoding="utf-8")
            print(f"[УСПІХ] Лог збережено у {log_path}")

    def _build_report(
        self,
        frame_w: int,
        frame_h: int,
        frames_processed: int,
        total_time: float,
        frame_fps_list: List[float],
        track_times_ms: List[float],
        detect_times_ms: List[float],
        classify_times_ms: List[float],
        draw_times_ms: List[float],
        active_counts: List[int],
        total_created: int,
        total_candidates: int,
        total_confirmed: int,
        skipped_duplicates: int,
        removed_reasons: dict,
        timeline_rows: List[str],
    ) -> str:
        fps_arr = np.array(frame_fps_list) if frame_fps_list else np.array([0.0])
        trk_arr = np.array(track_times_ms) if track_times_ms else np.array([0.0])
        det_arr = np.array(detect_times_ms) if detect_times_ms else np.array([0.0])
        cls_arr = np.array(classify_times_ms) if classify_times_ms else np.array([0.0])
        drw_arr = np.array(draw_times_ms) if draw_times_ms else np.array([0.0])
        act_arr = np.array(active_counts) if active_counts else np.array([0])

        lines = [
            f"=== AI OBJECT TRACKER PERFORMANCE REPORT ({self.target_label.upper()}) ===",
            f"Video: {self.video_path.name} ({frame_w}x{frame_h})",
            f"Config: detect_interval={self.detect_interval}, conf={self.conf_thresh:.2f}, "
            f"iou={self.iou_thresh:.2f}, ios={self.ios_thresh:.2f}, pad={self.pad_ratio:.2f}",
            f"Frames processed: {frames_processed} | Total time: {total_time:.2f}s",
            "",
            "--- FPS ---",
            f"Avg: {np.mean(fps_arr):.2f} | Median: {np.median(fps_arr):.2f} | "
            f"P10 (lows): {np.percentile(fps_arr, 10):.2f} | "
            f"Min: {np.min(fps_arr):.2f} | Max: {np.max(fps_arr):.2f}",
            "",
            "--- TIMING BREAKDOWN (ms per call) ---",
            f"Trackers update : avg={np.mean(trk_arr):.2f} ms | max={np.max(trk_arr):.2f} ms",
            f"MOG2 detect     : avg={np.mean(det_arr):.2f} ms | max={np.max(det_arr):.2f} ms",
            f"RF classify     : avg={np.mean(cls_arr):.2f} ms | max={np.max(cls_arr):.2f} ms",
            f"Draw + write    : avg={np.mean(drw_arr):.2f} ms | max={np.max(drw_arr):.2f} ms",
            "",
            "--- DETECTION & TRACKER STATS ---",
            f"Active trackers : avg={np.mean(act_arr):.2f} | max={np.max(act_arr)}",
            f"Total created   : {total_created}",
            f"MOG2 candidates : {total_candidates} -> RF confirmed: {total_confirmed} (skipped duplicates: {skipped_duplicates})",
            f"Removed reasons : lost_kcf={removed_reasons['lost_kcf']}, oob={removed_reasons['oob']}, "
            f"static={removed_reasons['static']}, dedup={removed_reasons['dedup']}",
            "",
            "--- TIMELINE (every 30 frames) ---",
            "frame | fps   | active | track_ms | detect_ms | classify_ms | candidates",
            *timeline_rows,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print("=== ВИБІР РЕЖИМУ ДЕТЕКЦІЇ (AI PIPELINE) ===")
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

    video_path = Path("sources/test.mp4")

    tracker = ObjectTracker(
        video=video_path,
        model=model_path,
        target_label=box_label,
        target_width=800,
        confidence_threshold=0.72,
        detect_interval=5,
    )
    tracker.process(
        debug=True,
        log_path=Path(f"logs/AI_tracker_log_{mode_name}.txt"),
        output_video=Path(f"logs/AI_output_{mode_name}.mp4"),
    )