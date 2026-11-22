import random
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PATCH_SIZE = (64, 64)

def extract_features(patch: cv2.typing.MatLike) -> np.ndarray:
    """Метод вилучення вектору числових ознак із вирізаного фрагмента зображення"""
    resized = cv2.resize(patch, PATCH_SIZE, interpolation=cv2.INTER_AREA)

    # 1. Колірні статистики в просторі LAB
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    mean, std = cv2.meanStdDev(lab)
    lab_features = np.concatenate([mean.flatten(), std.flatten()])

    # 2. Щільність контурів за детектором Кенні
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.array([np.mean(edges > 0)], dtype=np.float32)

    # 3. Спрощений HOG (гістограма кутів градієнта)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ang, bins=8, range=(0, 360), weights=mag)
    hist = hist / (np.sum(hist) + 1e-6)

    return np.concatenate([lab_features, edge_density, hist]).astype(np.float32)


def _boxes_intersect(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    """Перевіряє, чи перетинаються дві рамки формату (x1, y1, x2, y2)"""
    return not (
        box_a[2] <= box_b[0]
        or box_a[0] >= box_b[2]
        or box_a[3] <= box_b[1]
        or box_a[1] >= box_b[3]
    )


def _find_subdir(parent: Path, candidates: tuple[str, ...]) -> Path:
    """Шукає першу існуючу підпапку зі списку можливих назв"""
    for name in candidates:
        path = parent / name
        if path.exists():
            return path
    raise FileNotFoundError(f"У папці {parent} не знайдено жодної з підпапок: {candidates}")


def load_yolo_split(split_dir: Path, max_images: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Зчитує папку датасету формату YOLO, вирізає об'єкти машин та випадковий фон"""
    img_dir = _find_subdir(split_dir, ("images", "image"))
    lbl_dir = _find_subdir(split_dir, ("labels", "valid", "annotations"))

    image_paths = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if max_images is not None:
        image_paths = image_paths[:max_images]

    X: list[np.ndarray] = []
    y: list[int] = []

    for img_path in image_paths:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        car_boxes: list[tuple[int, int, int, int]] = []

        # 1. Вирізаємо всі машини за координатами з файлу розмітки
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, xc, yc, bw, bh = map(float, parts[:5])
                x1 = max(0, int((xc - bw / 2) * w))
                y1 = max(0, int((yc - bh / 2) * h))
                x2 = min(w, int((xc + bw / 2) * w))
                y2 = min(h, int((yc + bh / 2) * h))

                # Відсікаємо занадто дрібні об'єкти (шум)
                if (x2 - x1) < 16 or (y2 - y1) < 16:
                    continue

                car_boxes.append((x1, y1, x2, y2))
                patch = frame[y1:y2, x1:x2]
                X.append(extract_features(patch))
                y.append(1)  # 1 = машина

        # 2. Генеруємо таку ж кількість патчів фону без перетину з машинами
        bg_needed = max(1, len(car_boxes))
        bg_collected = 0
        attempts = 0

        while bg_collected < bg_needed and attempts < bg_needed * 10:
            attempts += 1
            crop_size = random.randint(48, min(160, h, w))
            rx1 = random.randint(0, w - crop_size)
            ry1 = random.randint(0, h - crop_size)
            cand = (rx1, ry1, rx1 + crop_size, ry1 + crop_size)

            if any(_boxes_intersect(cand, car_box) for car_box in car_boxes):
                continue

            patch = frame[ry1 : ry1 + crop_size, rx1 : rx1 + crop_size]
            X.append(extract_features(patch))
            y.append(0)  # 0 = фон
            bg_collected += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main() -> None:
    """Основна функція збору даних, навчання моделі RandomForest та її збереження"""
    dataset_root = Path("dataset")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("Завантаження train...")
    # Якщо в датасеті 10 000+ фото, для швидкого тесту можна задати max_images=1000
    X_train, y_train = load_yolo_split(dataset_root / "train", max_images=None)
    print(f"Train зібрано: {len(y_train)} патчів (машин: {np.sum(y_train == 1)}, фону: {np.sum(y_train == 0)})")

    print("Формування валідаційної вибірки...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    print(f"Valid зібрано: {len(y_val)} патчів")

    # Блок ініціалізації та навчання класифікатора
    print("Навчання RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    print("\nРезультати на валідації:")
    print(classification_report(y_val, y_pred, target_names=["background", "car"]))

    out_path = models_dir / "rf_people.joblib"
    joblib.dump({"model": rf, "classes": ["background", "car"]}, out_path)
    print(f"[УСПІХ] Модель збережено у {out_path}")


if __name__ == "__main__":
    main()