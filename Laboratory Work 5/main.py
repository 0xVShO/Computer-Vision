import argparse
from pathlib import Path
import cv2
import joblib
import numpy as np
import customtkinter as ctk
from skimage.segmentation import slic, mark_boundaries
from features import FeaturesLib
from dataset_form import DatasetForm

class Identificator:
    def __init__(self, img_name: str, root: ctk.CTk, n_segments: int = 600, compactness: float = 15.0):
        """Ініціалізація об'єкту класу для експериментів з ДЗЗ"""
        self.DATASET_DIR = Path('./dataset')
        self.SOURCES_DIR = Path('./sources')
        self.MODELS_DIR = Path('./models')

        self.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.img_path = self.SOURCES_DIR / img_name
        self.image = cv2.imread(str(self.img_path))
        if self.image is None:
            raise FileNotFoundError(f"Файл {self.img_path} не знайдено!")

        self.root = root
        self.n_segments = n_segments
        self.compactness = compactness

        self.model_path = self.MODELS_DIR / 'rf_classifier.joblib'
        self.model = None
        self.classes = ['buildings', 'forest', 'grass', 'road']

        # Палітра кольорів (BGR) для візуалізації кожного класу
        self.class_colors = {
            'buildings': (0, 0, 255),    # Червоний
            'forest': (0, 128, 0),       # Темно-зелений
            'grass': (144, 238, 144),    # Світло-зелений
            'road': (203, 192, 255)      # Рожевий
        }

    def _segmentation(self, preprocessed_img: np.ndarray) -> np.ndarray:
        """Метод суперпіксельної сегментації SLIC (приліплює межі до країв об'єктів)"""
        rgb_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
        segments = slic(
            rgb_img,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=1,
            channel_axis=-1
        )
        return segments

    def _training(self, interactive: bool = False) -> None:
        """Навчання моделі: або через вікно розмітки, або одразу по готовому датасету"""
        dataset_form = DatasetForm(img_path=str(self.img_path), root=self.root)
        if interactive:
            print("[ІНФО] Запуск вікна ручної розмітки та навчання...")
            self.model = dataset_form.process()
        else:
            print("[ІНФО] Перенавчання моделі на існуючих зрізах у ./dataset/...")
            self.model = dataset_form.train_model()

    def _load_model(self) -> None:
        """Завантаження збереженої моделі з диску"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Збережену модель не знайдено! Спочатку запустіть скрипт із прапорцем --train"
            )
        saved_data = joblib.load(self.model_path)
        self.model = saved_data['model']
        self.classes = saved_data['classes']
        print(f"[УСПІХ] Завантажено модель для класів: {self.classes}")

    def _predict_features(self, preprocessed_img: np.ndarray, global_mask: np.ndarray, segments: np.ndarray):
        """Збирає ознаки та повертає ймовірності для всіх сегментів один раз, щоб перемикання було миттєвим"""
        if self.model is None:
            self._load_model()

        unique_labels = np.unique(segments)
        features_batch = []
        valid_labels = []
        otsu_ratios = []

        for label in unique_labels:
            seg_mask = (segments == label).astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(seg_mask)

            patch = preprocessed_img[y:y + h, x:x + w]
            local_seg_mask = seg_mask[y:y + h, x:x + w]
            otsu_patch = global_mask[y:y + h, x:x + w]

            seg_area = cv2.countNonZero(local_seg_mask)
            otsu_inside = cv2.countNonZero(cv2.bitwise_and(otsu_patch, otsu_patch, mask=local_seg_mask))
            otsu_ratio = otsu_inside / max(seg_area, 1)

            feats = FeaturesLib.extract_features(patch, local_seg_mask, otsu_patch)
            features_batch.append(feats)
            valid_labels.append(label)
            otsu_ratios.append(otsu_ratio)

        probas = self.model.predict_proba(np.array(features_batch))
        return valid_labels, probas, otsu_ratios

    def _render_class(self, target_class: str, valid_labels, probas, otsu_ratios, segments) -> np.ndarray:
        """Фільтрує передбачення за обраним класом та накладає відповідний колір"""
        target_mask = np.zeros(segments.shape[:2], dtype=np.uint8)
        
        if target_class not in self.classes:
            return self.image.copy()
            
        class_idx = self.classes.index(target_class)
        color = self.class_colors.get(target_class, (255, 255, 255))

        for label, probs, otsu_ratio in zip(valid_labels, probas, otsu_ratios):
            pred_idx = int(np.argmax(probs))
            
            # Базова перевірка: чи збігається передбачений клас з тим, який ми хочемо намалювати
            is_target = (pred_idx == class_idx)
            
            # Для будівель залишаємо сувору фільтрацію за маскою Отсу з вашого оригінального коду
            if target_class == 'buildings':
                is_target = is_target and (probs[class_idx] >= 0.72) and (otsu_ratio >= 0.15)
            # Для інших класів поріг імовірності можна зробити стандартним
            else:
                is_target = is_target and (probs[class_idx] >= 0.5)

            if is_target:
                target_mask[segments == label] = 255

        result_img = self.image.copy()
        overlay = result_img.copy()
        overlay[target_mask == 255] = color

        return cv2.addWeighted(result_img, 0.5, overlay, 0.5, 0)

    def process(self, mode: str = "identify") -> None:
        """Основний метод виконання програми залежно від режиму запуску"""
        if mode == "annotate":
            self._training(interactive=True)
            return

        if mode == "train":
            self._training(interactive=False)
            return

        # Режим розпізнавання ('identify')
        preprocessed = FeaturesLib.preprocess(self.image)
        global_mask = FeaturesLib.mask_creation(preprocessed)
        segments = self._segmentation(preprocessed)

        slic_preview = (mark_boundaries(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB), segments) * 255).astype(np.uint8)
        slic_preview = cv2.cvtColor(slic_preview, cv2.COLOR_RGB2BGR)

        cv2.imshow("1. Otsu Gradient Mask", global_mask)
        cv2.imshow("2. SLIC Superpixels", slic_preview)

        print("[ІНФО] Вилучення ознак та класифікація сегментів...")
        valid_labels, probas, otsu_ratios = self._predict_features(preprocessed, global_mask, segments)

        current_class = 'buildings'
        print("\n=== ІНТЕРАКТИВНЕ ПЕРЕМИКАННЯ КЛАСІВ ===")
        print("b - Будівлі (Червоний)")
        print("f - Ліс (Темно-зелений)")
        print("g - Трава (Світло-зелений)")
        print("r - Дороги (Сірий)")
        print("ESC - Вихід")

        # Нескінченний цикл для відстеження натискань клавіш
        while True:
            identified_img = self._render_class(current_class, valid_labels, probas, otsu_ratios, segments)
            cv2.imshow("3. Detected Objects", identified_img)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # Натиснуто ESC
                break
            elif key == ord('b'):
                current_class = 'buildings'
            elif key == ord('f'):
                current_class = 'forest'
            elif key == ord('g'):
                current_class = 'grass'
            elif key == ord('r'):
                current_class = 'road'

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Модуль сегментації та ідентифікації об'єктів ДЗЗ")
    parser.add_argument("--img", type=str, default="google_maps.png")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--segments", type=int, default=600)
    parser.add_argument("--compactness", type=float, default=15.0)

    args = parser.parse_args()

    # Создаем окно CTk только тогда, когда оно реально нужно (в режиме --annotate)
    root = ctk.CTk() if args.annotate else None

    identificator = Identificator(
        img_name=args.img,
        root=root,
        n_segments=args.segments,
        compactness=args.compactness
    )

    if args.annotate:
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        identificator.process(mode="annotate")
    elif args.train:
        identificator.process(mode="train")
    else:
        identificator.process(mode="identify")
