import cv2
import numpy as np

class FeaturesLib:
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Метод первинної обробки зображення"""
        # Блок створення об'єкту методу CLAHE
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Блок конвертації зображення до простору LAB
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)

        # Блок фільтрації шумів білатеральним фільтром та підсилення контрасту методом CLAHE
        l_filtered = cv2.bilateralFilter(l_channel, d=9, sigmaColor=75, sigmaSpace=75)
        l_clahe = CLAHE.apply(l_filtered)

        merged_lab = cv2.merge((l_clahe, a_channel, b_channel))  # Збираємо канали назад з оновленою яскравістю
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)  # Повертаємо відфільтроване зображення у моделі BGR

    @staticmethod
    def mask_creation(preprocessed_img: np.ndarray) -> np.ndarray:
        """Метод створення масок для об'єктів шляхом використання класичних методів комп'ютерного зору"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        grayscaled_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
        morph_grad = cv2.morphologyEx(grayscaled_img, cv2.MORPH_GRADIENT, kernel)
        _, binary_img = cv2.threshold(
            morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Блок морфологічного очищення маски від шумів та заповнення розривів
        morph_open = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def extract_features(
        patch: np.ndarray,
        valid_mask: np.ndarray,
        otsu_patch: np.ndarray
    ) -> np.ndarray:
        """Вилучення вектору числових ознак із кольорового зрізу та масок"""
        pixel_count = cv2.countNonZero(valid_mask)
        if pixel_count == 0:
            return np.zeros(8, dtype=np.float32)

        # 1. Щільність білих контурів маски всередині об'єкта
        edges_inside = cv2.bitwise_and(otsu_patch, otsu_patch, mask=valid_mask)
        edge_density = cv2.countNonZero(edges_inside) / pixel_count

        # 2. Колірні статистики в просторі LAB (перетворюємо з (3, 1) у плоский список із 3 чисел)
        lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        mean_lab, std_lab = cv2.meanStdDev(lab_patch, mask=valid_mask)
        mean_l, mean_a, mean_b = mean_lab.flatten()
        std_l, std_a, std_b = std_lab.flatten()

        # 3. Коефіцієнт заповнення прямокутника (Extent)
        h, w = valid_mask.shape
        extent = pixel_count / float(w * h)

        return np.array([
            edge_density,
            mean_l, mean_a, mean_b,
            std_l, std_a, std_b
        ], dtype=np.float32)