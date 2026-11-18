import os
import glob
import cv2
import joblib
import numpy as np
import customtkinter as ctk
from sklearn.ensemble import RandomForestClassifier
from features import FeaturesLib

class DatasetForm:
    def __init__(self, img_path: str, root: ctk.CTk):
        """Ініціалізація початкових параметрів об'єкту"""
        raw_image = cv2.imread(img_path)
        if raw_image is None:
            raise FileNotFoundError(f"Фото за шляхом {img_path} не знайдено!")

        # Одразу готуємо препроцесингове фото та глобальну маску для синхронної нарізки
        self.image = FeaturesLib.preprocess(raw_image)
        self.global_mask = FeaturesLib.mask_creation(self.image)

        self.clusters = ['buildings', 'forest', 'grass', 'road']
        for cluster in self.clusters:
            os.makedirs(f'./dataset/{cluster}', exist_ok=True)

        self.cluster_choice = f'./dataset/{self.clusters}'
        self.model = None

        # Блок ініціалізації вікна налаштувань
        self.root = root
        if self.root is not None:
            self.root.geometry('400x400')
            self.root.title('Параметри формування датасету')

        # Блок ініціалізації допоміжних атрибутів
        self.current_points = []
        self.saved_polygons = []

    def _interface_init(self) -> None:
        """Метод ініціалізації вікна налаштувань"""
        self.clusters_menu = ctk.CTkOptionMenu(
            master=self.root,
            values=self.clusters,
            command=lambda choice: setattr(self, "cluster_choice", f'./dataset/{choice}'),
        )
        self.clusters_menu.pack(pady=15)
        self.clusters_menu.set(self.clusters)

        self.start_btn = ctk.CTkButton(
            master=self.root,
            text="Почати ручну розмітку",
            command=self._start_annotation
        )
        self.start_btn.pack(pady=10)

        self.seg_btn = ctk.CTkButton(
            master=self.root,
            text="Сегментувати великі області (64x64)",
            command=lambda: self._segment(self.cluster_choice)
        )
        self.seg_btn.pack(pady=10)

        self.train_btn = ctk.CTkButton(
            master=self.root,
            text="Навчити модель та завершити",
            fg_color="green",
            command=self._finish_and_train
        )
        self.train_btn.pack(pady=20)

    def _start_annotation(self) -> None:
        """Обгортка для приховування меню Tkinter та запуску OpenCV"""
        self.root.withdraw()
        self._cv_window_init()

    def _cv_window_init(self) -> None:
        """Ініціалізація вікна фото з об'єктами для формування датасету"""
        cv2.namedWindow('Оброблене зображення')
        cv2.setMouseCallback('Оброблене зображення', self._click_handle)

        while True:
            display_img = self.image.copy()

            for poly in self.saved_polygons:
                pts_array = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_img, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

            for i in range(len(self.current_points)):
                cv2.circle(display_img, self.current_points[i], 4, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display_img, self.current_points[i - 1], self.current_points[i], (0, 0, 255), 2)

            if len(self.current_points) == 4:
                cv2.line(display_img, self.current_points[3], self.current_points[0], (0, 0, 255), 2)

            cv2.imshow('Оброблене зображення', display_img)
            key = cv2.waitKey(15) & 0xFF

            if key == 13 and len(self.current_points) == 4:  # Клавіша Enter
                pts = np.array(self.current_points, np.int32)
                poly_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(poly_mask, [pts], 255)

                # Накладаємо маску полігону на фото та на глобальну маску контурів
                masked_img = cv2.bitwise_and(self.image, self.image, mask=poly_mask)
                masked_otsu = cv2.bitwise_and(self.global_mask, self.global_mask, mask=poly_mask)

                x, y, w, h = cv2.boundingRect(pts)
                cropped_img = masked_img[y:y + h, x:x + w]
                cropped_otsu = masked_otsu[y:y + h, x:x + w]

                # Зберігаємо як два звичайних файли без альфа-каналу (прозорості)
                next_idx = len(glob.glob(f'{self.cluster_choice}/slice_*.png'))
                img_save_path = f'{self.cluster_choice}/slice_{next_idx}.png'
                otsu_save_path = f'{self.cluster_choice}/otsu_{next_idx}.png'

                cv2.imwrite(filename=img_save_path, img=cropped_img)
                cv2.imwrite(filename=otsu_save_path, img=cropped_otsu)
                print(f"[УСПІХ] Збережено {img_save_path} та {otsu_save_path}")

                self.saved_polygons.append(list(self.current_points))
                self.current_points.clear()

            elif key == 8:  # Клавіша Backspace
                self.current_points.clear()

            elif key == 27:  # Клавіша Esc
                self.current_points.clear()
                break

        cv2.destroyAllWindows()
        self.root.deiconify()

    def _click_handle(self, event, x, y, flags, param) -> None:
        """Обробник натискання лівої клавіші миші"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.current_points) < 4:
            self.current_points.append((x, y))

    def _segment(self, folder_src: str) -> None:
        """Метод синхронної сегментації великих однорідних областей та їхніх масок на патчи 64x64"""
        files = glob.glob(os.path.join(folder_src, "slice_*.png"))
        if not files:
            print(f"[УВАГА] У папці {folder_src} немає файлів для сегментації.")
            return

        patch_size = 64
        patch_counter = len(glob.glob(os.path.join(folder_src, "patch_*.png")))

        for file in files:
            bgr_img = cv2.imread(file)
            otsu_path = file.replace("slice_", "otsu_")
            otsu_img = cv2.imread(otsu_path, cv2.IMREAD_GRAYSCALE)

            if bgr_img is None or otsu_img is None:
                continue

            h, w = bgr_img.shape[:2]
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    patch_bgr = bgr_img[y:y + patch_size, x:x + patch_size]
                    patch_otsu = otsu_img[y:y + patch_size, x:x + patch_size]

                    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

                    if cv2.countNonZero(gray) >= (patch_size * patch_size * 0.95):
                        cv2.imwrite(os.path.join(folder_src, f"patch_{patch_counter}.png"), patch_bgr)
                        cv2.imwrite(os.path.join(folder_src, f"patch_otsu_{patch_counter}.png"), patch_otsu)
                        patch_counter += 1

        print(f"[УСПІХ] Сегментацію завершено. У {folder_src} тепер {patch_counter} фрагментів (64x64).")

    def train_model(self) -> RandomForestClassifier:
        """Збирає ознаки з датасету за допомогою FeaturesLib та навчає модель"""
        X_features = []
        y_labels = []

        for label_idx, cluster in enumerate(self.clusters):
            folder = f'./dataset/{cluster}'
            patches = glob.glob(os.path.join(folder, "patch_*.png"))
            # Відфільтровуємо файли масок, щоб брати лише кольорові фото
            patches = [p for p in patches if "patch_otsu_" not in p]
            files = patches if patches else glob.glob(os.path.join(folder, "slice_*.png"))

            for file_path in files:
                bgr_patch = cv2.imread(file_path)
                otsu_path = file_path.replace("patch_", "patch_otsu_") if "patch_" in file_path else file_path.replace("slice_", "otsu_")
                otsu_patch = cv2.imread(otsu_path, cv2.IMREAD_GRAYSCALE)

                if bgr_patch is None or otsu_patch is None:
                    continue

                # Маска самого об'єкта (відсікає чорне тло навколо полігону)
                gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
                _, valid_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                features = FeaturesLib.extract_features(bgr_patch, valid_mask, otsu_patch)
                X_features.append(features)
                y_labels.append(label_idx)

        if not X_features:
            raise ValueError("Датасет порожній! Спочатку розмітьте об'єкти на фото.")

        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(np.array(X_features), np.array(y_labels))

        os.makedirs('./models', exist_ok=True)
        joblib.dump({'model': clf, 'classes': self.clusters}, './models/rf_classifier.joblib')
        print(f"[УСПІХ] Модель навчено на {len(X_features)} зразках та збережено у ./models/rf_classifier.joblib")

        self.model = clf
        return clf

    def _finish_and_train(self) -> None:
        """Обробник кнопки навчання та закриття вікна"""
        self.train_model()
        self.root.quit()
        self.root.destroy()

    def process(self) -> RandomForestClassifier:
        """Основний публічний метод, який повертає об'єкт навченої моделі"""
        self._interface_init()
        self.root.mainloop()
        return self.model
