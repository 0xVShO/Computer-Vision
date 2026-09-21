import tkinter as tk
from tkinter import filedialog
import numpy as np
import json

class Engine:
    def __init__(self, root):
        '''Ініціалізація головного вікна'''
        self.root = root
        self.root.title("3D Engine Simulation")

        # Налаштування розмірів канвасу
        self.width = 800
        self.height = 800

        # Створення канвасу для відображення графіки
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.btn = tk.Button(self.root, text="Load Object", command=self.load_object)
        self.btn.pack()
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Початкові параметри кутів повротів об'єкта (ініціалізація кутів обертання навколо осей X, Y та Z)
        self.angle_x = 0
        self.angle_y = 0

        # Початкові параметри переміщення об'єкта (ініціалізація координат переміщення об'єкта у 3D просторі)
        self.last_x = 0
        self.last_y = 0

        # Останні координати миші (ініціалізація останніх координат миші для обробки подій перетягування)
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Початкові параметри масштабування об'єкта (ініціалізація коефіцієнта масштабування об'єкта)
        self.scale = 1.0

        # Початкові параметри швидкості переміщення об'єкта(ініціалізація швидкостей переміщення об'єкта у 3D просторі)
        self.velocity_x = 0
        self.velocity_y = 0

        # Початкові параметри швидкості обертання об'єкта(ініціалізація швидкостей обертання об'єкта у 3D просторі)
        self.velocity_rx = 0
        self.velocity_ry = 0

        self.distance = 500.0 # Відстань від камери до об'єкта для проекції
        self.FOV = 100 # Масштабний коефіцієнт для проекції об'єкта на 2D площину канвасу
        self.friction = 0.85 # Коефіцієнт тертя для плавного зменшення швидкості об'єкта

        # Прив'язка обробників подій миші до канвасу
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_scroll)
        self.canvas.bind("<Button-2>", self.on_middle_press)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)

    def load_object(self):
        '''Завантаження об'єкта з JSON файлу'''
        filepath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filepath:
            with open(filepath, 'r') as file:
                data = json.load(file)
                obj_vertices = data.get("obj_vertices", [])
                obj_edges = data.get("obj_edges", [])
                self.btn.destroy()
                self.scene_init(obj_vertices, obj_edges)
                self.render()  # Виклик методу render для відображення завантаженого об'єкта

    def scene_init(self, obj_vertices=None, obj_edges=None):
        '''Ініціалізація об'єктів сцени'''
        self.obj_vertices = np.array(obj_vertices) # Матриця вершин об'єкта
        self.obj_edges = np.array(obj_edges) # Матриця ребер об'єкта

        self.axes_vertices = np.array([[0, 0, 0, 1], [0.5, 0, 0, 1], [0, 0.5, 0, 1], [0, 0, 0.5, 1]]) # Матриця вершин координатних осей
        self.axes_edges = np.array([[0, 1], [0, 2], [0, 3]]) # Матриця ребер координатних осей

    def update_physics(self):
        '''Оновлення параметрів об'єкта на основі його швидкостей та тертя (симуляція інерції та плавного зменшення швидкості об'єкта)'''
        # Оновлення координат об'єкта на основі його швидкостей переміщення
        self.last_x += self.velocity_x
        self.last_y += self.velocity_y

        # Оновлення кутів обертання об'єкта на основі його швидкостей обертання
        self.angle_x += self.velocity_rx
        self.angle_y += self.velocity_ry

        # Застосування тертя для плавного зменшення швидкостей переміщення об'єкта
        self.velocity_x *= self.friction
        self.velocity_y *= self.friction

        # Застосування тертя для плавного зменшення швидкостей обертання об'єкта
        self.velocity_rx *= self.friction
        self.velocity_ry *= self.friction

    def get_transfer_matrix(self, dx, dy, dz):
        '''Обчислення матриці переносу об'єкта у 3D просторі'''
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [dx, dy, dz, 1]
        ])

    def get_rotation_matrix(self, angle_x, angle_y):
        '''Обчислення матриці обертання навколо осей X, Y та Z'''
        rx = np.array([ # Матриця обертання навколо осі X
            [1, 0, 0, 0],
            [0, np.cos(angle_x), np.sin(angle_x), 0],
            [0, -np.sin(angle_x), np.cos(angle_x), 0],
            [0, 0, 0, 1]
        ])

        ry = np.array([ # Матриця обертання навколо осі Y
            [np.cos(angle_y), 0, np.sin(angle_y), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y), 0],
            [0, 0, 0, 1]
        ])

        rz = np.eye(4) # Матриця обертання навколо осі Z (ідентична матриця, оскільки обертання навколо Z відсутнє на рівні приладу вводу)

        # Повертаємо матрицю, що є результатом композиції матриць обертання навколо осей Z, Y та X відповідно
        return rz @ ry @ rx

    def get_scale_matrix(self, scale):
        '''Обчислення матриці масштабування об'єкта у 3D просторі'''
        return np.array([
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0],
            [0, 0, 0, 1]
        ])

    def project_vertex(self, vertex):
        '''Перетворює 3D координати вершини у 2D координати канвасу'''
        x_2d = (vertex[0] / (vertex[2] / self.distance + 1)) * self.FOV + self.width / 2
        y_2d = (-vertex[1] / (vertex[2] / self.distance + 1)) * self.FOV + self.height / 2
    
        return x_2d, y_2d

    def render(self):
        '''Відображення об'єкта на канвасі'''
        self.update_physics()  # Оновлення параметрів об'єкта перед відображенням
        self.canvas.delete("all")  # Очищення канвасу перед новим відображенням

        # Отримання матриць трансформацій (масштабування, обертання та переносу)
        scale_matrix = self.get_scale_matrix(self.scale)
        rotation_matrix = self.get_rotation_matrix(self.angle_x, self.angle_y)
        transfer_matrix = self.get_transfer_matrix(self.last_x, self.last_y, 0)

        # Комбінування матриць трансформацій у одну матрицю
        transformation_matrix = scale_matrix @ rotation_matrix @ transfer_matrix

        # Застосування матриці трансформацій до вершин об'єкта
        transformed_vertices = self.obj_vertices @ transformation_matrix

        # Застосування матриці трансформацій до вершин координатних осей
        transformed_axes_vertices = self.axes_vertices @ transformation_matrix

        # Відображення ребер об'єкта на канвасі
        for edge in self.obj_edges:
            start_vertex = transformed_vertices[edge[0]]
            end_vertex = transformed_vertices[edge[1]]

            # Проекція 3D координат на 2D площину канвасу
            x1, y1 = self.project_vertex(start_vertex)
            x2, y2 = self.project_vertex(end_vertex)

            # Малювання лінії між двома вершинами ребра
            self.canvas.create_line(x1, y1, x2, y2, fill="black")

        axis_colors = ["red", "blue", "green"]

        for axe, edge in enumerate(self.axes_edges):
            start_vertex = transformed_axes_vertices[edge[0]]
            end_vertex = transformed_axes_vertices[edge[1]]

            # Проекція 3D координат на 2D площину канвасу
            x1, y1 = self.project_vertex(start_vertex)
            x2, y2 = self.project_vertex(end_vertex)

            # Малювання лінії між двома вершинами ребра координатних осей
            self.canvas.create_line(x1, y1, x2, y2, fill=axis_colors[axe])
        
        self.root.after(16, self.render)  # Виклик методу render через 16 мс для створення ефекту анімації
    
    def on_mouse_press(self, event):
        '''Обробка натискання клавіші миші'''
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        self.velocity_rx = 0
        self.velocity_ry = 0

    def on_mouse_drag(self, event):
        '''Обробка перетягування клавіші миші'''
        dx = (event.x - self.last_mouse_x) * 0.01
        dy = (event.y - self.last_mouse_y) * 0.01

        self.velocity_rx = dy
        self.velocity_ry = -dx

        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_mouse_scroll(self, event):
        '''Обробка прокручування колеса миші'''
        if event.delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1

    def on_middle_press(self, event):
        '''Обробка натискання середньої клавіші миші'''
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

        self.velocity_x = 0
        self.velocity_y = 0

    def on_middle_drag(self, event):
        '''Обробка перетягування середньої клавіші миші'''
        dx = (event.x - self.last_mouse_x) * 0.01
        dy = (event.y - self.last_mouse_y) * 0.01

        self.velocity_x = dx
        self.velocity_y = -dy

        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

root = tk.Tk()
engine = Engine(root)
root.mainloop()  # Запуск головного циклу обробки подій