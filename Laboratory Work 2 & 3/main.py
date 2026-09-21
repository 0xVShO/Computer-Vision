import cv2
import numpy as np
import customtkinter as ctk
from time import sleep
import os

class WatershedRectPipeline:
    '''Загальний клас для інтерфейсу та пайплайну виконання детекції будівель'''
    def __init__(self, img_src, root):
        self.img = cv2.imread(img_src) 
        if self.img is None:
            raise FileNotFoundError(f"Фото за шляхом {img_src} не знайдено!")
            
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) # Зчитування фото у форматі відтінків сірого
        
        self.attr = {} # Словник аттрибутів параметрів для пайплайну
        self.labels = {} # Словник для написів над повзунками зміни параметрів
        self.show_mask = ctk.BooleanVar(value=False) # Флаг для переключення режиму перегляду між маскою та результатами детекції

        # Блок налаштування інтерфейсу
        self.root = root
        self.root.title("Watershed & Rectangular Detection")
        self.root.geometry("450x950")

    def param_init(self, **kwargs):
        '''Метод динамічної ініціалізації параметрів'''
        self.attr.update(kwargs)

    def interface_init(self):
        '''Метод ініціалізації інтерфейсу налаштувань та перегляду результатів'''
        checkbox = ctk.CTkCheckBox(
            master=self.root, 
            text="Показувати маску (Watershed Regions)", 
            variable=self.show_mask, 
            command=self.pipeline
        )
        checkbox.pack(pady=(15, 10))
        
        for key, values in self.attr.items():
            minimum, value, maximum, step_val = values
            label = ctk.CTkLabel(master=self.root, text=f"{key}: {value}")
            label.pack(pady=(2, 0))
            self.labels[key] = label

            # Динамічне створення слайдерів для зміни відповідних параметрів
            slider = ctk.CTkSlider(
                master=self.root,
                from_=minimum,
                to=maximum,
                number_of_steps=int((maximum - minimum) / step_val) if step_val > 0 else None,
                command=lambda val, k=key: self.on_slider_change(k, val)
            )
            slider.pack(pady=(2, 5))
            slider.set(value)

    def on_slider_change(self, key, value):
        '''Метод обробки при зміні значення на слайдері'''
        valid_value = self.value_check(key, value)
        self.attr[key][1] = valid_value
        self.labels[key].configure(text=f"{key}: {valid_value}")
        self.pipeline()

    def value_check(self, key, raw_value):
        '''Метод перевірки корректності введених через слайдер даних'''
        minimum, _, maximum, step = self.attr[key]
        if step == 0: return raw_value
        steps_count = round((raw_value - minimum) / step)
        snapped_value = minimum + (steps_count * step)
        snapped_value = max(minimum, min(snapped_value, maximum))
        return int(snapped_value) if isinstance(step, int) else snapped_value

    def pipeline(self):
        '''Основний метод для виконання загального алгоритму детекції будівель'''
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
        blurred = cv2.bilateralFilter(self.gray_img, blur_k, 50, 50) # Використання Білатерального фільтру для усунення зайвих шумів зі збереженням контрасту границь об'єктів
        clahe = cv2.createCLAHE(clipLimit=self.attr['_2_clahe_clip'][1], tileGridSize=(8, 8)) # Коррекція контрасту границь об'єктів
        contrasted = clahe.apply(blurred)

        thresh_block_size = max(3, thresh_block | 1) # Ядор адаптивної бінарізації
        thresh = cv2.adaptiveThreshold( # Блок адаптивної бінарізації 
            contrasted, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            thresh_block_size, thresh_C
        )

        # Морфологія та монолітна маска
        edges = cv2.Canny(thresh, 50, 150) # Детекція границь Кенні
        k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, self.attr['_5_dilate_edges'][1] | 1),)*2)
        dilated_edges = cv2.dilate(edges, k_dilate, iterations=1)

        # Блок малювання знайдних границь об'єктів
        raw_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(self.gray_img)
        cv2.drawContours(solid_mask, raw_contours, -1, 255, -1)

        # Блок морфологічних операцій
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, self.attr['_6_erode_mask'][1] | 1),)*2) # Операція Ерозії
        separated_mask = cv2.erode(solid_mask, k_erode, iterations=1)

        # Ядра для морфологічних операцій
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, open_k | 1),)*2)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, close_k | 1),)*2)
        
        cleaned_mask = cv2.morphologyEx(separated_mask, cv2.MORPH_OPEN, k_open) # Морфологічна операція відкриття
        final_solid_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, k_close) # Морфологічна операція закриття

        # Сегментація методом вододілу
        sure_bg = cv2.dilate(final_solid_mask, np.ones((3,3), np.uint8), iterations=3)
        dist_transform = cv2.distanceTransform(final_solid_mask, cv2.DIST_L2, 5)
        
        ret, sure_fg = cv2.threshold(dist_transform, watershed_dist * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1 
        markers[unknown == 255] = 0 
        
        blurred_bgr = cv2.bilateralFilter(self.img, blur_k, 50, 50)
        markers = cv2.watershed(blurred_bgr, markers)

        if self.show_mask.get():
            display_mask = np.zeros_like(self.img)
            display_mask[markers == -1] = [0, 0, 255] 
            
            for i in range(2, ret + 1):
                color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
                display_mask[markers == i] = color
                
            cv2.imshow("Detection Result", display_mask)
            cv2.waitKey(1)
            return

        # Векторизація прямокутниками (minAreaRect)
        output_img = self.img.copy()
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
                # Отримуємо мінімальний прямокутник
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                
                if width == 0 or height == 0:
                    continue
                    
                rect_area = width * height
                extent = area / rect_area

                # Відсіюємо сміття за параметром Extent
                if extent >= min_extent:
                    valid_buildings_count += 1
                    # Отримуємо 4 кути прямокутника і малюємо його
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(output_img, [box], 0, (0, 0, 255), 2)

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Знайдено будівель (Watershed + Rectangles): {valid_buildings_count}")
        
        cv2.imshow("Detection Result", output_img)
        cv2.waitKey(1)

if __name__ == "__main__":
    bing_img_src = r"./sources/bing_maps.png"
    google_img_src = r"./sources/google_maps.png"
    
    ctk.set_appearance_mode("Dark")
    root = ctk.CTk()
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Оберіть фото для детекції:\n1) Bing Maps\n2) Google Maps")
        choice = input("Номер варіанту: ")
        
        if choice in ['1', '2']:
            img_src = bing_img_src if choice == '1' else google_img_src
            GUI = WatershedRectPipeline(img_src=img_src, root=root) # Ініціаліація екземпляру класу інтерфейсу
            
            GUI.param_init( # Ініціалізація необхіднхи параметрів для пайплайну
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
                _11_watershed_dist=[1, 35, 99, 1] 
            )
            break
        else:
            print("Спробуйте ще раз!")
            sleep(1)
            
    GUI.interface_init() # Ініціалізація інтерфейсу
    GUI.pipeline() # Запуск загального алгоритму детекції будівель
    root.mainloop()