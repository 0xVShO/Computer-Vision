import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
import psutil
import os

class ObjectTracker:
    def __init__(self, tracker_type: Literal['KCF', 'CSRT', 'MOSSE'] = 'KCF'):
        """Ініціалізація об'єкта відстежувача з вибором алгоритму відстеження."""

        # Ініціалізація історії для зберігання даних про FPS, використання пам'яті та успішність відстеження
        self.fps_history = []
        self.memory_usage_history = []
        self.success_history = []

        # Ініціалізація словника методів відстеження для підтримки різних алгоритмів
        self.tracker_methods = {
            'KCF': cv2.TrackerKCF.create,
            'CSRT': cv2.TrackerCSRT.create,
            'MOSSE': cv2.legacy.TrackerMOSSE.create
        }

        # Ініціалізація типу відстежувача та створення об'єкта відстежувача
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker(tracker_type)
        self.bbox = None
        self.process = psutil.Process(os.getpid())


    def _create_tracker(self, tracker_type, tracker_params=None) -> cv2.Tracker:
        """Створення об'єкта відстежувача на основі вказаного типу відстеження."""
        if tracker_type not in self.tracker_methods:
            raise ValueError(f"Непідтримуваний тип відстежувача: {tracker_type}")

        if tracker_params is None:
            tracker_params = {}

        try:
            return self.tracker_methods[tracker_type](**tracker_params)
        except TypeError as e:
            raise ValueError(f"Неправильні параметри для {tracker_type}: {e}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Попередня обробка кадру для покращення якості зображення перед відстеженням."""
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_frame)

        l_channel = CLAHE.apply(l_channel)
        lab_frame = cv2.merge((l_channel, a_channel, b_channel))
        frame_enhanced = cv2.cvtColor(lab_frame, cv2.COLOR_LAB2BGR)

        blurred = cv2.bilateralFilter(frame_enhanced, 5, 75, 75)
        return blurred
        

    def pipeline(self, video_path: str) -> None:
        """Основний алгоритм відстеження об'єкта в відео."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = self._preprocess_frame(frame)
            self.bbox = cv2.selectROI("Select object", frame, False)
            self.tracker.init(frame, self.bbox)
        else:
            print("Error: Unable to read the video file.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self._preprocess_frame(frame)
            timer = cv2.getTickCount()

            success, new_bbox = self.tracker.update(frame)
            if success:
                self.bbox = new_bbox
                p1 = (int(self.bbox[0]), int(self.bbox[1]))
                p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                self.success_history.append(True)
            else:
                self.success_history.append(False)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            self.fps_history.append(fps)

            mem_mb = self.process.memory_info().rss / (1024 * 1024)
            self.memory_usage_history.append(mem_mb)

            cv2.imshow("Object Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self._analyze()
        return

    def _analyze(self) -> None:
        """Аналіз результатів відстеження та відображення графіків."""
        if not self.fps_history:
            print("Помилка: Немає даних для аналізу. Відео не було оброблено.")
            return

        total_frames = len(self.fps_history)
        mean_fps = sum(self.fps_history) / total_frames
        success_rate = (sum(self.success_history) / len(self.success_history)) * 100
        mean_memory = sum(self.memory_usage_history) / len(self.memory_usage_history)
        peak_memory = max(self.memory_usage_history)

        print("\n" + "="*50)
        print(f"Детальний аналіз: Алгоритм {self.tracker_type}")
        print("="*50)
        print(f"Всього оброблено кадрів: {total_frames}")
        print(f"Середня кількість кадрів за секунду (FPS): {mean_fps:.2f} кадрів/с")
        print(f"Точність відстеження об'єкту:  {success_rate:.2f}%")
        print(f"Середнє використання ОП: {mean_memory:.2f} МБ")
        print(f"Пікове використання ОП: {peak_memory:.2f} МБ")
        print("="*50 + "\n")

        frames = range(1, total_frames + 1)
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(f"Аналіз ефективності алгоритму: {self.tracker_type}", fontsize=16, fontweight='bold')

        axs[0].plot(frames, self.fps_history, color='blue', label='Поточний FPS')
        axs[0].axhline(mean_fps, color='red', linestyle='--', label=f'Середній FPS ({mean_fps:.2f})')
        axs[0].set_title('Швидкість обробки кадрів (FPS)')
        axs[0].set_ylabel('FPS')
        axs[0].grid(True, linestyle=':', alpha=0.7)
        axs[0].legend()

        axs[1].plot(frames, self.memory_usage_history, color='green', label='Використання RAM (МБ)')
        axs[1].axhline(mean_memory, color='red', linestyle='--', label=f'Середнє ОЗУ ({mean_memory:.2f} МБ)')
        axs[1].set_title('Споживання оперативної пам\'яті')
        axs[1].set_ylabel('Мегабайти (МБ)')
        axs[1].grid(True, linestyle=':', alpha=0.7)
        axs[1].legend()

        success_numeric = [1 if status else 0 for status in self.success_history]
        axs[2].step(frames, success_numeric, color='purple', where='mid')
        axs[2].fill_between(frames, success_numeric, step="mid", color='purple', alpha=0.2)
        axs[2].set_title('Статус виявлення об\'єкта (1 - знайдено, 0 - втрачено)')
        axs[2].set_xlabel('Номер кадру')
        axs[2].set_ylabel('Статус')
        axs[2].set_yticks([0, 1])
        axs[2].set_yticklabels(['Втрачено', 'Знайдено'])
        axs[2].grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.show()

if __name__ == "__main__":
    tracker = ObjectTracker(tracker_type='CSRT')
    tracker.pipeline(r'sources/V_7.mp4')