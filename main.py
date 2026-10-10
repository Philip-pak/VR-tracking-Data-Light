"""
Скрипт для визуализации данных VR-трекинга поверх видео
Основная задача: синхронизировать данные пространственного положения головы и рук (XR-Robotics)
с кадрами видео и наложить их в виде траекторий и меток.
"""

import argparse  # Для парсинга аргументов командной строки (пути к файлам, макс. секунды)
import json # Для чтения построчного JSON (формат JSON Lines в файле трекинга)
import re # Для работы с регулярными выражениями (извлечение чисел из строк с запятыми)
from pathlib import Path

import cv2 # OpenCV
import numpy as np # NumPy
from PIL import Image, ImageDraw, ImageFont  #Pillow: рендеринг растровой графики и поддержка кириллицы


# Регулярное выражение для извлечения чисел из строк трекинга
# где числа записаны с десятичной запятой (например: "-0,08517717,-0,02745685,...").
NUMBER_RE = re.compile(r"[-+]?\d+,\d+|[-+]?\d+(?:\.\d+)?")


def load_font(size):
    """
    Загрузка шрифта Windows, поддерживающего кириллицу
    Проверяет стандартные пути системных шрифтов, при отсутствии возвращает стандартный шрифт Pillow.
    """
    font_paths = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
    ]
    for path in font_paths:
        if path.exists():
            return ImageFont.truetype(str(path), size) # возвращаем первый найденный шрифт
    return ImageFont.load_default() # если шрифта нет, используем стандартный шрифт без кириллицы 
# Инициализация шрифтов разных размеров для меток и легенды
FONT_BIG = load_font(28) #заголовок
FONT_MEDIUM = load_font(21) #метки объектов
FONT_SMALL = load_font(18) 


def parse_pose(text):
    """
    Извлекает координаты [x, y, z] из строки позы (pose) формата XR-Robotics.
    Заменяет запятую на точку для корректного преобразования в float.
    """
    if not text:
        return None
    # Находим все числа в строке и приводим к типу float
    values = [float(x.replace(",", ".")) for x in NUMBER_RE.findall(text)]
    if len(values) < 3: # Проверяем, что удалось извлечь хотя бы 3 координаты (X, Y, Z)
        return None
    return values[:3]  # Возвращаем только пространственные координаты X, Y, Z

# 2. ЗАГРУЗКА И ПАРСИНГ ФАЙЛА ТРЕКИНГА (JSON LINES)

def load_tracking(path):
    """
    Чтение файла трекинга в формате JSON Lines (каждая строка — отдельный JSON-объект с таймстампом).
    Возвращает словарь с массивами numpy для времени, головы и ладоней рук.
    """
    # Список временных для сбора данных
    times = [] 
    head = []
    left = []
    right = []

    with open(path, "r", encoding="utf-8") as file: # Открытие файла трекинга в кодировке UTF-8
        # Читаем первую строку для определения начального времени (timestamp)
        first_row = json.loads(file.readline())
        start_time_ns = int(first_row["timeStampNs"]) # Начинаем читать данные от сюда, чтобы не захватить не нужное

        # Построчно считываем весь файл трекинга
        for line in file:
            row = json.loads(line)
            # Переводим наносекунды в секунды относительно начала записи
            times.append((int(row["timeStampNs"]) - start_time_ns) / 1_000_000_000)

            # Извлекаем положение головы (если данных нет, заполняем NaN (массив))
            head.append(parse_pose(row.get("Head", {}).get("pose")) or [np.nan, np.nan, np.nan])

            # Извлекаем суставы рук
            hand = row.get("Hand", {})
            left_joints = hand.get("leftHand", {}).get("HandJointLocations", [])
            right_joints = hand.get("rightHand", {}).get("HandJointLocations", [])

            # Сустав 0 в схеме XR-Robotics соответствует ладони (Palm) сустав с индексом 0 — это ладонь
            left.append(parse_pose(left_joints[0]["p"]) if left_joints else [np.nan, np.nan, np.nan])
            right.append(parse_pose(right_joints[0]["p"]) if right_joints else [np.nan, np.nan, np.nan])

    return {  # Превращаем списки в оптимизированные массивы NumPy для быстрого поиска и обработки
        "time": np.array(times),
        "head": np.array(head, dtype=float),
        "left": np.array(left, dtype=float),
        "right": np.array(right, dtype=float),
    }


def make_mapper(tracking, width, height):
    """               
                    =======================================================================================   
                        Работаем только по X/Y координатам (игнорируем Z), так как видео — это 2D-плоскость.
                    =======================================================================================

    Создает функцию-преобразователь мировых координат VR (X/Y) в пиксели кадра видео (x/y).
    
    Поскольку точной калибровки нет,
    используется приближенная нормализация по перцентилям (2% и 98%), чтобы отсечь редкие выбросы 
    и равномерно распределить область движения по площади кадра.
    """
    all_points = np.vstack([tracking["head"], tracking["left"], tracking["right"]])
    all_points = all_points[np.isfinite(all_points).all(axis=1)]

    # Вычисляем границы движения по перцентилям для стабильности
    x_min, y_min = np.percentile(all_points[:, [0, 1]], 2, axis=0)
    x_max, y_max = np.percentile(all_points[:, [0, 1]], 98, axis=0)

    def to_pixel(point):
        #Внутренняя функция для конвертации одной трехмерной точки в пиксели (px, py)
        x, y, _z = point
        # Интерполируем VR-координаты в размеры кадра видео (с отступами по краям)
        px = int(np.interp(x, [x_min, x_max], [width * 0.15, width * 0.85]))
        py = int(np.interp(y, [y_min, y_max], [height * 0.85, height * 0.15]))
        # Ограничиваем координаты в пределах разрешения видео
        px = int(np.clip(px, 0, width - 1))
        py = int(np.clip(py, 0, height - 1))
        return px, py

    return to_pixel


def draw_label(frame, text, point, color):
    """Рисует точку текущего положения объекта и возвращает параметры для подписи."""
    x, y = point
    # Рисуем заполненный кружок с белой обводкой для контраста, радиус 10 пикселей
    cv2.circle(frame, (x, y), 10, color, -1, cv2.LINE_AA)
    # Рисуем внешнюю белую обводку радиусом 12 пикселей для максимальной контрастности на любом фоне
    cv2.circle(frame, (x, y), 12, (255, 255, 255), 2, cv2.LINE_AA)
    return text, (x + 14, y - 28), color, FONT_MEDIUM


def draw_texts(frame, items):
    """Отрисовка русского текста на кадре OpenCV с помощью библиотеки Pillow (для поддержки кириллицы)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for text, position, color, font in items:
        # OpenCV использует формат BGR, Pillow использует RGB, конвертируем цвета обратно
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_legend(frame):
    """Создает полупрозрачную подложку и возвращает элементы легенды (описания цветов)."""
    overlay = frame.copy()
    # Рисуем черный прямоугольник подложки
    cv2.rectangle(overlay, (20, 20), (600, 210), (0, 0, 0), -1)
    # Накладываем подложку с прозрачностью 60% через addWeighted
    cv2.addWeighted(overlay, 0.60, frame, 0.38, 0, frame)
    return [
        ("Примерное наложение VR-трекинга", (35, 32), (255, 255, 255), FONT_BIG),
        ("Желтый: голова", (35, 70), (70, 220, 255), FONT_MEDIUM),
        ("Оранжевый: левая ладонь", (35, 98), (70, 150, 255), FONT_MEDIUM),
        ("Синий: правая ладонь", (35, 124), (255, 170, 70), FONT_MEDIUM),
        ("Автор: Пахомов Филипп", (35, 150), (255, 255, 255), FONT_MEDIUM),
        ("Telegram: @Filipp_SMM", (35, 180), (255, 160, 255), FONT_MEDIUM),
    ]


# ==============================================================================
# 4. ОСНОВНОЙ ЦИКЛ РЕНДЕРИНГА ВИДЕО
# ==============================================================================

def render_overlay(video_path, tracking_path, output_path, max_seconds=None):
    """Основная функция рендеринга: покадрово читает видео, синхронизирует с трекингом и сохраняет результат."""
    # Загружаем данные трекинга
    tracking = load_tracking(tracking_path) 
    # Открываем видеофайл с помощью OpenCV
    video = cv2.VideoCapture(str(video_path))  # 2. Открываем видеофайл с помощью OpenCV
    if not video.isOpened():
        raise RuntimeError(f"Не удалось открыть видеофайл: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 30 # 3. Извлекаем ключевые параметры видео (FPS, разрешение, общее число кадров)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_to_render = frame_count
    if max_seconds is not None: # Ограничиваем количество кадров для рендеринга, если задан параметр max_seconds
        frames_to_render = min(frame_count, int(max_seconds * fps))

    # Создаем директорию для результатов и инициализируем VideoWriter для записи MP4
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
     
    # Инициализируем функцию проецирования координат
    to_pixel = make_mapper(tracking, width, height)
    
    # Словари для хранения истории шлейфа (хвоста) траектории
    trails = {"head": [], "left": [], "right": []}
    colors = {
        "head": (70, 220, 255),
        "left": (70, 150, 255),
        "right": (255, 170, 70),
    }

    frame_index = 0
    preview_saved = False
    
    # Цикл обработки каждого кадра видео
    while frame_index < frames_to_render:
        ok, frame = video.read()
        if not ok:
            break

        # Определяем текущее время видео в секундах
        seconds = frame_index / fps
        # синхронизация. Находим ближайший индекс в массиве данных трекинга по времени
        tracking_index = np.searchsorted(tracking["time"], seconds, side="right") - 1
        tracking_index = int(np.clip(tracking_index, 0, len(tracking["time"]) - 1))

        # Обновляем координаты и добавляем их в хвост трейла
        for name in ("head", "left", "right"):
            point = tracking[name][tracking_index]
            if np.isfinite(point).all():
                trails[name].append(to_pixel(point))
            # Ограничиваем длину хвоста (последние 90 точек ≈ 3 секунды при 30 fps)
            trails[name] = trails[name][-90:]

        # Рисуем линии шлейфа траектории движения
        for name, trail in trails.items():
            if len(trail) > 1:
                cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, colors[name], 3, cv2.LINE_AA)

        # Собираем текстовые метки и легенду для отрисовки
        text_items = []
        if trails["head"]:
            text_items.append(draw_label(frame, "голова", trails["head"][-1], colors["head"]))
        if trails["left"]:
            text_items.append(draw_label(frame, "левая", trails["left"][-1], colors["left"]))
        if trails["right"]:
            text_items.append(draw_label(frame, "правая", trails["right"][-1], colors["right"]))
        text_items.extend(draw_legend(frame))
        
        # Отрисовываем весь текст и маркеры на кадре
        draw_texts(frame, text_items)

        # Записываем кадр в результирующий видеофайл
        writer.write(frame)

        # Сохраняем первый обработанный кадр
        if not preview_saved:
            preview_path = output_path.parent / "preview.jpg"
            cv2.imwrite(str(preview_path), frame)
            preview_saved = True

        frame_index += 1

        # Вывод прогресса в консоль каждые 300 кадров
        if frame_index % 200 == 0:
            print(f"Обработано кадров: {frame_index}/{frames_to_render}")

    # Освобождаем ресурсы видеопотоков
    video.release()
    writer.release()
    print(f"Готово! Видео сохранено в: {output_path}")


def main():
    """Настройка аргументов командной строки и запуск обработки."""
    parser = argparse.ArgumentParser(description="Приближенное наложение VR-трекинга на видео.")
    parser.add_argument("--video", default="data/CameraRecord_20260505_165740.mp4")
    parser.add_argument("--tracking", default="data/trackingData_20260505_165740.txt")
    parser.add_argument("--output", default="results/overlay_full.mp4")
    parser.add_argument("--max-seconds", type=float, default=None, help="Обработать только первые N секунд для быстрой проверки.")
    args = parser.parse_args()

    render_overlay(Path(args.video), Path(args.tracking), Path(args.output), args.max_seconds)

if __name__ == "__main__":
    main()