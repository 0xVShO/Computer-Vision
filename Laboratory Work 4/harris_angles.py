import cv2
import numpy as np
import customtkinter as ctk
from time import sleep
import os

class WatershedRectPipeline:
    '''Загальний клас для інтерфейсу, збереженого пайплайну детекції будівель та вертикального порівняння точок'''
    def __init__(self, img1_src, img2_src, root):
        self.img1 = cv2.imread(img1_src) 
        self.img2 = cv2.imread(img2_src)
        
        if self.img1 is None:
            raise FileNotFoundError(f"Фото за шляхом {img1_src} не знайдено!")
        if self.img2 is None:
            raise FileNotFoundError(f"Фото за шляхом {img2_src} не знайдено!")
            
        # Зчитування обох фото у форматі відтінків сірого (строго як в оригіналі)
        self.gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        
        self.attr = {} # Словник атрибутів параметрів для пайплайну
        self.labels = {} # Словник для написів над повзунками зміни параметрів
        self.show_mask = ctk.BooleanVar(value=False) # Прапорець перемикання на маску Watershed
        self.show_matching = ctk.BooleanVar(value=True) # Прапорець перемикання на порівняння особливих точок

        # Блок налаштування інтерфейсу
        self.root = root
        self.root.title("Watershed & Vertical Feature Matching")
        self.root.geometry("460x980")

    def param_init(self, **kwargs):
        '''Метод динамічної ініціалізації параметрів'''
        self.attr.update(kwargs)

    def interface_init(self):
        '''Метод ініціалізації інтерфейсу налаштувань та перегляду результатів'''
        checkbox_mask = ctk.CTkCheckBox(
            master=self.root, 
            text="Показувати маску (Watershed Regions)", 
            variable=self.show_mask, 
            command=self.pipeline
        )
        checkbox_mask.pack(pady=(10, 5))

        checkbox_match = ctk.CTkCheckBox(
            master=self.root, 
            text="Режим порівняння особливих точок (Зверху/Знизу)", 
            variable=self.show_matching, 
            command=self.pipeline
        )
        checkbox_match.pack(pady=(5, 5))

        self.prob_label = ctk.CTkLabel(
            master=self.root, 
            text="Ймовірність P: розрахунок...", 
            font=("Arial", 14, "bold"),
            text_color="#00FF7F"
        )
        self.prob_label.pack(pady=(2, 5))

        scroll_frame = ctk.CTkScrollableFrame(master=self.root, width=420, height=780)
        scroll_frame.pack(pady=5, padx=10, fill="both", expand=True)
        
        for key, values in self.attr.items():
            minimum, value, maximum, step_val = values
            label = ctk.CTkLabel(master=scroll_frame, text=f"{key}: {value}")
            label.pack(pady=(2, 0))
            self.labels[key] = label

            # Динамічне створення повзунків для зміни відповідних параметрів
            slider = ctk.CTkSlider(
                master=scroll_frame,
                from_=minimum,
                to=maximum,
                number_of_steps=int((maximum - minimum) / step_val) if step_val > 0 else None,
                command=lambda val, k=key: self.on_slider_change(k, val)
            )
            slider.pack(pady=(2, 5))
            slider.set(value)

    def on_slider_change(self, key, value):
        '''Метод обробки при зміні значення на повзунку'''
        valid_value = self.value_check(key, value)
        self.attr[key][1] = valid_value
        self.labels[key].configure(text=f"{key}: {valid_value}")
        self.pipeline()

    def value_check(self, key, raw_value):
        '''Метод перевірки коректності введених через повзунок даних'''
        minimum, _, maximum, step = self.attr[key]
        if step == 0: return raw_value
        steps_count = round((raw_value - minimum) / step)
        snapped_value = minimum + (steps_count * step)
        snapped_value = max(minimum, min(snapped_value, maximum))
        return int(snapped_value) if isinstance(step, int) else snapped_value

    def apply_color_correction_vis(self, bgr_img):
        '''Неінвазивна корекція кольору (для вимог Лаби №2), яка використовується лише для фонової візуалізації та не змінює основний пайплайн детекції'''
        gamma = self.attr['_15_color_gamma_pct'][1] / 100.0
        if gamma == 1.0:
            return bgr_img.copy()
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(bgr_img, table)

    def run_original_detection(self, bgr_img, gray_img, is_google=False):
        '''Оригінальний пайплайн детекції будівель без жодних змін у логіці та фільтрах'''
        if is_google:
            # Трюк для ортофотоплану: створюємо штучні "стіни" (контурний об'єм) 
            # через морфологічний градієнт та високочастотне підсилення
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph_grad = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
            
            # Змішуємо оригінальний сірий кадр з градієнтом країв (ефект "псевдо-фасадів")
            gray_img = cv2.addWeighted(gray_img, 0.8, morph_grad, 0.2, 0)

        # Отримання базових параметрів
        blur_k = self.attr['_1_bilateral_blur'][1]
        thresh_block = self.attr['_3_thresh_block'][1]
        thresh_C = self.attr['_4_thresh_C'][1]
        open_k = self.attr['_7_morph_open'][1]
        close_k = self.attr['_8_morph_close'][1]
        min_area = self.attr['_9_min_area'][1]
        
        # Extent (Прямокутність)
        min_extent = self.attr['_10_min_extent_pct'][1] / 100.0
        watershed_dist = self.attr['_11_watershed_dist'][1] / 100.0

        # Підготовка та поріг
        blurred = cv2.bilateralFilter(gray_img, blur_k, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=self.attr['_2_clahe_clip'][1], tileGridSize=(8, 8))
        contrasted = clahe.apply(blurred)

        thresh_block_size = max(3, thresh_block | 1)
        thresh = cv2.adaptiveThreshold(
            contrasted, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            thresh_block_size, thresh_C
        )

        # Морфологія та монолітна маска
        edges = cv2.Canny(thresh, 50, 150)
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, self.attr['_5_dilate_edges'][1] | 1),)*2)
        dilated_edges = cv2.dilate(edges, k_dilate, iterations=1)

        # Блок малювання знайдених границь об'єктів
        raw_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(gray_img)
        cv2.drawContours(solid_mask, raw_contours, -1, 255, -1)

        # Блок морфологічних операцій
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, self.attr['_6_erode_mask'][1] | 1),)*2)
        separated_mask = cv2.erode(solid_mask, k_erode, iterations=1)

        # Ядра для морфологічних операцій
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, open_k | 1),)*2)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, close_k | 1),)*2)
        
        cleaned_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_OPEN, k_open)
        final_solid_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, k_close)

        # Сегментація методом вододілу
        sure_bg = cv2.dilate(final_solid_mask, np.ones((3,3), np.uint8), iterations=3)
        dist_transform = cv2.distanceTransform(final_solid_mask, cv2.DIST_L2, 5)
        
        ret, sure_fg = cv2.threshold(dist_transform, watershed_dist * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1 
        markers[unknown == 255] = 0 
        
        blurred_bgr = cv2.bilateralFilter(bgr_img, blur_k, 50, 50)
        markers = cv2.watershed(blurred_bgr, markers)

        # Маска для режиму перегляду Watershed
        display_mask = np.zeros_like(bgr_img)
        display_mask[markers == -1] = [0, 0, 255] 
        for i in range(2, ret + 1):
            color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
            display_mask[markers == i] = color

        # Векторизація прямокутниками (minAreaRect) та збір маски знайдених будівель
        output_img = self.apply_color_correction_vis(bgr_img)
        buildings_roi_mask = np.zeros_like(gray_img)
        valid_buildings_count = 0
        
        for marker_id in range(2, ret + 1):
            obj_mask = np.zeros_like(final_solid_mask)
            obj_mask[markers == marker_id] = 255
            
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            
            if area > min_area:
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                
                if width == 0 or height == 0:
                    continue
                    
                rect_area = width * height
                extent = area / rect_area

                # Відсіюємо хибні об'єкти за параметром Extent
                if extent >= min_extent:
                    valid_buildings_count += 1
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(output_img, [box], 0, (0, 0, 255), 2)
                    # Записуємо знайдену будівлю в маску об'єктів для подальшого пошуку точок
                    cv2.drawContours(buildings_roi_mask, [box], 0, 255, -1)

        return contrasted, output_img, display_mask, buildings_roi_mask, valid_buildings_count

    def stack_vertical_and_fit(self, top_img, bottom_img, max_total_height=920):
        '''Допоміжний метод: вирівнює дві картинки за шириною, склеює їх зверху донизу та масштабує під екран'''
        h1, w1 = top_img.shape[:2]
        h2, w2 = bottom_img.shape[:2]
        
        target_w = max(w1, w2)
        if w1 != target_w:
            top_img = cv2.resize(top_img, (target_w, int(h1 * target_w / w1)))
        if w2 != target_w:
            bottom_img = cv2.resize(bottom_img, (target_w, int(h2 * target_w / w2)))
            
        stacked = np.vstack((top_img, bottom_img))
        
        total_h, total_w = stacked.shape[:2]
        if total_h > max_total_height:
            scale = max_total_height / total_h
            stacked = cv2.resize(stacked, (int(total_w * scale), max_total_height))
            
        return stacked

    def match_and_draw_vertical(self, img_top, gray_top, mask_top, img_bot, gray_bot, mask_bot):
        '''Пошук особливих точок, оцінка ймовірності та спеціалізоване малювання ліній зверху донизу без обрізання країв'''
        roi_pad = self.attr['_12_roi_padding'][1]
        ratio_thresh = self.attr['_13_lowe_ratio'][1] / 100.0
        ransac_thresh = self.attr['_14_ransac_thresh'][1]

        # 1. Розширюємо маску знайдених будівель, щоб захопити кути дахів
        k_pad = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, roi_pad | 1),)*2)
        search_mask_top = cv2.dilate(mask_top, k_pad, iterations=1)
        search_mask_bot = cv2.dilate(mask_bot, k_pad, iterations=1)

        # 2. Детекція точок та генерація дескрипторів (SIFT)
        sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=15)
        kp_top, des_top = sift.detectAndCompute(gray_top, search_mask_top)
        kp_bot, des_bot = sift.detectAndCompute(gray_bot, search_mask_bot)

        valid_object_matches = []
        outlier_matches = []

        if des_top is not None and des_bot is not None and len(kp_top) >= 4 and len(kp_bot) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = bf.knnMatch(des_top, des_bot, k=2)

            good_matches = []
            for m_n in raw_matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

            # Відсіювання хибних збігів за допомогою гомографії RANSAC
            if len(good_matches) >= 4:
                src_pts = np.float32([kp_top[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_bot[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                _, ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(ransac_thresh))
                if ransac_mask is not None:
                    inliers = [good_matches[i] for i in range(len(good_matches)) if ransac_mask[i][0] == 1]
                else:
                    inliers = good_matches
            else:
                inliers = good_matches

            # Невеликий допуск (3px) для кутів, що лежать безпосередньо на контурі прямокутника будівлі
            eval_mask_top = cv2.dilate(mask_top, np.ones((5, 5), np.uint8), iterations=1)
            eval_mask_bot = cv2.dilate(mask_bot, np.ones((5, 5), np.uint8), iterations=1)

            for m in inliers:
                x1, y1 = int(round(kp_top[m.queryIdx].pt[0])), int(round(kp_top[m.queryIdx].pt[1]))
                x2, y2 = int(round(kp_bot[m.trainIdx].pt[0])), int(round(kp_bot[m.trainIdx].pt[1]))

                in_obj1 = (0 <= y1 < eval_mask_top.shape[0] and 0 <= x1 < eval_mask_top.shape[1]) and (eval_mask_top[y1, x1] > 0)
                in_obj2 = (0 <= y2 < eval_mask_bot.shape[0] and 0 <= x2 < eval_mask_bot.shape[1]) and (eval_mask_bot[y2, x2] > 0)

                if in_obj1 and in_obj2:
                    valid_object_matches.append(m)
                else:
                    outlier_matches.append(m)

        total_pts = len(valid_object_matches) + len(outlier_matches)
        obj_pts = len(valid_object_matches)
        prob = (obj_pts / total_pts) if total_pts > 0 else 0.0

        # 3. Кастомна вертикальна склейка (Зверху - Знімок 1, Знизу - Знімок 2) з перерахунком координат точок
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bot.shape[:2]
        target_w = max(w1, w2)

        scale_x1, scale_y1 = target_w / w1, target_w / w1
        scale_x2, scale_y2 = target_w / w2, target_w / w2

        top_resized = cv2.resize(img_top, (target_w, int(h1 * scale_y1))) if w1 != target_w else img_top.copy()
        bot_resized = cv2.resize(img_bot, (target_w, int(h2 * scale_y2))) if w2 != target_w else img_bot.copy()

        top_h = top_resized.shape[0]
        canvas = np.vstack((top_resized, bot_resized))

        # Масштабуємо полотно по висоті, щоб воно гарантовано вмістилося в екран монітора
        max_screen_h = 920
        total_h, total_w = canvas.shape[:2]
        final_scale = (max_screen_h / total_h) if total_h > max_screen_h else 1.0
        if final_scale < 1.0:
            canvas = cv2.resize(canvas, (int(total_w * final_scale), max_screen_h))

        # Малюємо лінії збігу поверх відмасштабованого вертикального полотна (лінії будуть чіткими та тонкими)
        # Спочатку червоні (фон), потім зелені (будівлі)
        for m in outlier_matches:
            pt1_raw = kp_top[m.queryIdx].pt
            pt2_raw = kp_bot[m.trainIdx].pt
            pt1 = (int(pt1_raw[0] * scale_x1 * final_scale), int(pt1_raw[1] * scale_y1 * final_scale))
            pt2 = (int(pt2_raw[0] * scale_x2 * final_scale), int((pt2_raw[1] * scale_y2 + top_h) * final_scale))
            cv2.circle(canvas, pt1, 3, (0, 0, 255), -1)
            cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)
            cv2.line(canvas, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

        for m in valid_object_matches:
            pt1_raw = kp_top[m.queryIdx].pt
            pt2_raw = kp_bot[m.trainIdx].pt
            pt1 = (int(pt1_raw[0] * scale_x1 * final_scale), int(pt1_raw[1] * scale_y1 * final_scale))
            pt2 = (int(pt2_raw[0] * scale_x2 * final_scale), int((pt2_raw[1] * scale_y2 + top_h) * final_scale))
            cv2.circle(canvas, pt1, 4, (0, 255, 0), -1)
            cv2.circle(canvas, pt2, 4, (0, 255, 0), -1)
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

        return canvas, obj_pts, total_pts, prob

    def pipeline(self):
        '''Основний метод для виконання алгоритму на обох знімках та вертикального відображення результатів'''
        contr1, out1, mask_vis1, roi_mask1, count1 = self.run_original_detection(self.img1, self.gray_img1, is_google=False)
        contr2, out2, mask_vis2, roi_mask2, count2 = self.run_original_detection(self.img2, self.gray_img2, is_google=True)

        match_canvas, obj_pts, total_pts, prob = self.match_and_draw_vertical(
            out1, contr1, roi_mask1, 
            out2, contr2, roi_mask2
        )

        # Оновлення тексту в інтерфейсі та консолі
        color_hex = "#00FF7F" if prob >= 0.9 else "#FF6347"
        self.prob_label.configure(
            text=f"Ймовірність P = {prob:.3f} ({obj_pts}/{total_pts}) | Будівель: {count1} та {count2}",
            text_color=color_hex
        )

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Знайдено будівель (Зверху Bing / Знизу Google): {count1} / {count2}")
        print(f"Придатних точок у порівнянні: {total_pts} | Належать об'єктам: {obj_pts}")
        print(f"Ймовірність ідентифікації P: {prob:.4f} (Ціль >= 0.90)")

        if self.show_mask.get():
            stacked_masks = self.stack_vertical_and_fit(mask_vis1, mask_vis2)
            cv2.imshow("Detection Result (Top: Bing, Bottom: Google)", stacked_masks)
        elif self.show_matching.get():
            cv2.imshow("Detection Result (Top: Bing, Bottom: Google)", match_canvas)
        else:
            stacked_rects = self.stack_vertical_and_fit(out1, out2)
            cv2.imshow("Detection Result (Top: Bing, Bottom: Google)", stacked_rects)

        cv2.waitKey(1)

if __name__ == "__main__":
    bing_img_src = r"./sources/bing_maps.png"
    google_img_src = r"./sources/google_maps.png"
    
    ctk.set_appearance_mode("Dark")
    root = ctk.CTk()
    
    GUI = WatershedRectPipeline(img1_src=bing_img_src, img2_src=google_img_src, root=root)
    
    GUI.param_init(
        _1_bilateral_blur=[1, 11, 21, 2],         
        _2_clahe_clip=[1, 1, 10, 1],
        _3_thresh_block=[3, 63, 151, 2],
        _4_thresh_C=[-30, -24, 30, 1],
        _5_dilate_edges=[1, 3, 15, 2],       
        _6_erode_mask=[1, 1, 21, 2],         
        _7_morph_open=[1, 7, 51, 2],        
        _8_morph_close=[1, 5, 51, 2], 
        _9_min_area=[100, 200, 10000, 50],
        _10_min_extent_pct=[5, 15, 100, 5],
        _11_watershed_dist=[1, 35, 99, 1],
        _12_roi_padding=[1, 3, 31, 2],        
        _13_lowe_ratio=[50, 75, 95, 5],       
        _14_ransac_thresh=[1, 5, 20, 1],      
        _15_color_gamma_pct=[50, 100, 150, 5] 
    )
            
    GUI.interface_init()
    GUI.pipeline()
    root.mainloop()