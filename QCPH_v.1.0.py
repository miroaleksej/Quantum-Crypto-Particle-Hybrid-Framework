import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, dctn, idctn
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
import time
import warnings
from collections import Counter

# Подавляем предупреждения, чтобы не засорять вывод
warnings.filterwarnings('ignore', category=UserWarning)

class CERNHypercubeFramework:
    """
    Основной класс гибридной системы гиперкуба для обнаружения новых физических явлений в данных CERN.
    
    Реализует следующие компоненты:
    1. Построение гиперкуба данных
    2. Сжатие данных с адаптивным пороговым квантованием
    3. Градиентный анализ для выявления структур
    4. Спектральный анализ для обнаружения коллизий
    5. Многослойный анализ для повышения достоверности
    6. Физическая интерпретация и обнаружение новых частиц
    """
    
    def __init__(self, parameters=None, compression_epsilon=1e-4, gamma=0.5, num_levels=3):
        """
        Инициализация гибридной системы гиперкуба.
        
        Параметры:
        - parameters: список физических параметров (например, ['eta1', 'phi1', 'pT', 'eta2', 'phi2', 'mass'])
        - compression_epsilon: базовый порог для сжатия данных
        - gamma: параметр для адаптивного порога
        - num_levels: количество уровней детализации для многослойного анализа
        """
        self.parameters = parameters or ['eta1', 'phi1', 'pT', 'eta2', 'phi2', 'mass']
        self.n = len(self.parameters)
        self.compression_epsilon = compression_epsilon
        self.gamma = gamma
        self.num_levels = num_levels
        self.hypercubes = []
        self.compressed_hypercubes = []
        self.standard_model = None
        self.standard_diagrams = None
        self.anomalies = []
    
    def build_hypercube(self, events, num_bins_per_dim=50, min_vals=None, max_vals=None):
        """
        Построение гиперкуба данных из событий коллайдера.
        
        Параметры:
        - events: массив событий размера (m, n), где m - количество событий, n - количество параметров
        - num_bins_per_dim: количество ячеек по каждой координате
        - min_vals: минимальные значения для каждого параметра (опционально)
        - max_vals: максимальные значения для каждого параметра (опционально)
        
        Возвращает:
        - hypercube: гиперкуб данных размера (num_bins_per_dim, ..., num_bins_per_dim)
        """
        m = events.shape[0]  # Количество событий
        n = events.shape[1]  # Количество параметров
        
        # Определение диапазонов для каждого параметра
        if min_vals is None:
            min_vals = np.min(events, axis=0)
        if max_vals is None:
            max_vals = np.max(events, axis=0)
        
        # Создание гиперкуба
        bins = [np.linspace(min_vals[i], max_vals[i], num_bins_per_dim + 1) for i in range(n)]
        hypercube = np.zeros(tuple([num_bins_per_dim] * n), dtype=float)
        
        # Заполнение гиперкуба
        for i in range(m):
            indices = []
            for j in range(n):
                bin_idx = np.digitize(events[i, j], bins[j]) - 1
                bin_idx = min(max(0, bin_idx), num_bins_per_dim - 1)
                indices.append(bin_idx)
            hypercube[tuple(indices)] += 1
        
        # Нормализация (плотность событий)
        hypercube /= np.sum(hypercube)
        
        return hypercube
    
    def adaptive_compression(self, hypercube, level=1):
        """
        Адаптивное сжатие гиперкуба с использованием DCT и адаптивного порогового квантования.
        
        Параметры:
        - hypercube: гиперкуб данных для сжатия
        - level: уровень детализации (1 - базовый, выше - более детальный)
        
        Возвращает:
        - compressed: сжатое представление гиперкуба
        - stats: статистика сжатия (коэффициент сжатия, ошибка восстановления)
        """
        # Вычисление DCT
        transformed = dctn(hypercube, norm='ortho')
        
        # Оценка локальной плотности для адаптивного порога
        density = self.estimate_local_density(hypercube)
        
        # Создание карты пороговых значений
        threshold_map = self.create_threshold_map(density, level)
        
        # Адаптивное пороговое квантование
        thresholded = np.zeros_like(transformed)
        non_zero_count = 0
        total_elements = np.prod(hypercube.shape)
        
        for idx in np.ndindex(transformed.shape):
            if abs(transformed[idx]) > threshold_map[idx] * np.linalg.norm(transformed):
                thresholded[idx] = transformed[idx]
                non_zero_count += 1
        
        # Вычисление коэффициента сжатия
        compression_ratio = non_zero_count / total_elements
        
        # Восстановление данных для оценки ошибки
        restored = idctn(thresholded, norm='ortho')
        error = np.linalg.norm(hypercube - restored) / np.linalg.norm(hypercube)
        
        # Сохранение только ненулевых коэффициентов для эффективного хранения
        compressed = {
            'shape': hypercube.shape,
            'non_zero_indices': np.where(thresholded != 0),
            'non_zero_values': thresholded[thresholded != 0],
            'threshold_map': threshold_map,
            'level': level
        }
        
        stats = {
            'compression_ratio': compression_ratio,
            'reconstruction_error': error,
            'non_zero_count': non_zero_count,
            'total_elements': total_elements
        }
        
        return compressed, stats
    
    def estimate_local_density(self, hypercube, window_size=3):
        """
        Оценка локальной плотности данных для адаптивного сжатия.
        
        Параметры:
        - hypercube: гиперкуб данных
        - window_size: размер окна для локальной оценки
        
        Возвращает:
        - density: карта локальной плотности
        """
        n_dims = len(hypercube.shape)
        density = np.zeros_like(hypercube)
        padded = np.pad(hypercube, window_size//2, mode='constant')
        
        # Скользящее окно по всем измерениям
        for idx in np.ndindex(hypercube.shape):
            slices = tuple(slice(i, i + window_size) for i in idx)
            local_region = padded[slices]
            density[idx] = np.mean(local_region)
        
        # Нормализация
        density = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-10)
        return density
    
    def create_threshold_map(self, density, level=1):
        """
        Создание карты пороговых значений на основе плотности и уровня детализации.
        
        Параметры:
        - density: карта локальной плотности
        - level: уровень детализации
        
        Возвращает:
        - threshold_map: карта пороговых значений
        """
        # Базовый порог зависит от уровня детализации
        base_epsilon = self.compression_epsilon * (2 ** (self.num_levels - level))
        
        # Адаптивный порог: ниже в областях высокой плотности (гладкие области)
        # и выше в областях низкой плотности (возможные аномалии)
        threshold_map = base_epsilon * np.exp(-self.gamma * density)
        
        return threshold_map
    
    def restore_hypercube(self, compressed):
        """
        Восстановление гиперкуба из сжатого представления.
        
        Параметры:
        - compressed: сжатое представление гиперкуба
        
        Возвращает:
        - restored: восстановленный гиперкуб
        """
        # Создание пустого преобразованного гиперкуба
        transformed = np.zeros(compressed['shape'])
        
        # Заполнение ненулевых коэффициентов
        transformed[compressed['non_zero_indices']] = compressed['non_zero_values']
        
        # Обратное DCT
        restored = idctn(transformed, norm='ortho')
        
        return restored
    
    def compute_persistent_homology(self, hypercube, max_dim=2, metric='euclidean'):
        """
        Вычисление персистентной гомологии для гиперкуба данных.
        
        Параметры:
        - hypercube: гиперкуб данных
        - max_dim: максимальная размерность гомологий
        - metric: метрика для вычисления расстояний
        
        Возвращает:
        - diagrams: персистентные диаграммы
        """
        # Преобразование гиперкуба в облако точек
        points = self.hypercube_to_points(hypercube)
        
        # Вычисление персистентной гомологии
        diagrams = ripser(points, maxdim=max_dim, metric=metric)['dgms']
        
        return diagrams
    
    def hypercube_to_points(self, hypercube, num_points=1000):
        """
        Преобразование гиперкуба в облако точек для анализа персистентной гомологии.
        
        Параметры:
        - hypercube: гиперкуб данных
        - num_points: количество точек для генерации
        
        Возвращает:
        - points: облако точек
        """
        n_dims = len(hypercube.shape)
        
        # Нормализация гиперкуба к вероятностному распределению
        prob = hypercube / np.sum(hypercube)
        
        # Генерация индексов точек пропорционально плотности
        indices = np.random.choice(
            np.arange(np.prod(hypercube.shape)),
            size=num_points,
            p=prob.flatten()
        )
        
        # Преобразование линейных индексов в многомерные координаты
        points = np.array(np.unravel_index(indices, hypercube.shape)).T
        
        # Нормализация координат к [0, 1]
        points = points / np.array(hypercube.shape)
        
        return points
    
    def compute_betti_numbers(self, diagrams, threshold=0.1):
        """
        Вычисление чисел Бетти на основе персистентных диаграмм.
        
        Параметры:
        - diagrams: персистентные диаграммы
        - threshold: порог персистентности для учета особенности
        
        Возвращает:
        - betti: числа Бетти [b0, b1, ..., b_max_dim]
        """
        betti = []
        for dim, diagram in enumerate(diagrams):
            # Удаляем точку, соответствующую бесконечной компоненте (обычно первая точка)
            finite_points = diagram[:-1] if len(diagram) > 0 and np.isinf(diagram[-1][1]) else diagram
            
            # Считаем количество особенностей с персистентностью выше порога
            count = np.sum(finite_points[:, 1] - finite_points[:, 0] > threshold)
            betti.append(count)
        
        return betti
    
    def compute_wasserstein_distance(self, diagram1, diagram2, p=1):
        """
        Вычисление расстояния Вассерштейна между двумя персистентными диаграммами.
        
        Параметры:
        - diagram1, diagram2: персистентные диаграммы
        - p: степень для расстояния Вассерштейна
        
        Возвращает:
        - distance: расстояние Вассерштейна
        """
        # Если одна из диаграмм пустая, возвращаем большое значение
        if len(diagram1) == 0 or len(diagram2) == 0:
            return 100.0
        
        # Удаляем бесконечные точки
        diagram1 = diagram1[:-1] if np.isinf(diagram1[-1][1]) else diagram1
        diagram2 = diagram2[:-1] if np.isinf(diagram2[-1][1]) else diagram2
        
        # Вычисляем расстояние Вассерштейна
        try:
            # Используем встроенную функцию из persim, если доступна
            from persim import wasserstein
            return wasserstein(diagram1, diagram2, p=p)
        except:
            # Простая реализация для p=1
            if p == 1:
                distances = []
                for pt1 in diagram1:
                    min_dist = float('inf')
                    for pt2 in diagram2:
                        dist = np.linalg.norm(pt1 - pt2, ord=1)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
                
                for pt2 in diagram2:
                    min_dist = float('inf')
                    for pt1 in diagram1:
                        dist = np.linalg.norm(pt1 - pt2, ord=1)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
                
                return np.mean(distances) / 2
            else:
                # Для p != 1 используем простую евклидову метрику
                return np.mean([np.min([np.linalg.norm(pt1 - pt2) for pt2 in diagram2]) for pt1 in diagram1])
    
    def compute_anomaly_indicator(self, hypercube, standard_diagrams=None):
        """
        Вычисление индикатора аномалии на основе персистентной гомологии.
        
        Параметры:
        - hypercube: гиперкуб данных для анализа
        - standard_diagrams: эталонные персистентные диаграммы для стандартной модели
        
        Возвращает:
        - anomaly_indicator: значение индикатора аномалии
        - diagrams: персистентные диаграммы для данного гиперкуба
        """
        if standard_diagrams is None:
            if self.standard_diagrams is None:
                raise ValueError("Standard diagrams not set. Call set_standard_model first.")
            standard_diagrams = self.standard_diagrams
        
        # Вычисление персистентной гомологии
        diagrams = self.compute_persistent_homology(hypercube)
        
        # Вычисление расстояния Вассерштейна для каждого измерения
        wasserstein_distances = []
        max_dim = min(len(diagrams), len(standard_diagrams))
        
        for dim in range(max_dim):
            dist = self.compute_wasserstein_distance(diagrams[dim], standard_diagrams[dim])
            wasserstein_distances.append(dist)
        
        # Взвешенная сумма расстояний
        weights = [1.0, 0.8, 0.5]  # Веса для dim 0, 1, 2
        anomaly_indicator = sum(w * d for w, d in zip(weights[:max_dim], wasserstein_distances))
        
        return anomaly_indicator, diagrams
    
    def set_standard_model(self, standard_events, num_bins_per_dim=50):
        """
        Установка эталонной модели на основе стандартных событий.
        
        Параметры:
        - standard_events: события, соответствующие стандартной модели
        - num_bins_per_dim: количество ячеек по каждой координате
        """
        # Построение гиперкуба для стандартной модели
        self.standard_hypercube = self.build_hypercube(
            standard_events, 
            num_bins_per_dim=num_bins_per_dim
        )
        
        # Вычисление эталонных персистентных диаграмм
        self.standard_diagrams = self.compute_persistent_homology(self.standard_hypercube)
        
        # Вычисление эталонных собственных значений для спектрального анализа
        self.standard_eigenvalues = self.compute_correlation_eigenvalues(self.standard_hypercube)
    
    def compute_correlation_eigenvalues(self, hypercube):
        """
        Вычисление собственных значений матрицы корреляций для гиперкуба.
        
        Параметры:
        - hypercube: гиперкуб данных
        
        Возвращает:
        - eigenvalues: собственные значения матрицы корреляций
        """
        # Преобразование гиперкуба в облако точек
        points = self.hypercube_to_points(hypercube, num_points=5000)
        
        # Вычисление матрицы корреляций
        corr_matrix = np.corrcoef(points, rowvar=False)
        
        # Вычисление собственных значений
        eigenvalues, _ = np.linalg.eigh(corr_matrix)
        
        # Сортировка по убыванию
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        return eigenvalues
    
    def gradient_analysis(self, hypercube):
        """
        Градиентный анализ для выявления скрытых зависимостей в данных.
        
        Параметры:
        - hypercube: гиперкуб данных
        
        Возвращает:
        - gradients: градиенты по каждому направлению
        - linear_dependencies: обнаруженные линейные зависимости
        """
        n_dims = len(hypercube.shape)
        gradients = []
        
        # Вычисление градиентов по каждому измерению
        for dim in range(n_dims):
            grad = np.gradient(hypercube, axis=dim)
            gradients.append(grad)
        
        # Поиск линейных зависимостей
        linear_dependencies = []
        
        # Для каждой точки в гиперкубе
        for idx in np.ndindex(hypercube.shape):
            # Пропускаем краевые точки
            if any(i == 0 or i == size-1 for i, size in zip(idx, hypercube.shape)):
                continue
            
            # Собираем градиенты в точке
            grad_values = [gradients[dim][idx] for dim in range(n_dims)]
            
            # Нормализуем градиенты
            grad_norm = np.linalg.norm(grad_values)
            if grad_norm > 1e-10:
                grad_values = [g / grad_norm for g in grad_values]
                
                # Ищем линейные комбинации
                for i in range(n_dims):
                    for j in range(i+1, n_dims):
                        # Проверяем, пропорциональны ли градиенты
                        ratio = grad_values[i] / (grad_values[j] + 1e-10)
                        if abs(ratio) > 0.1 and abs(ratio) < 10:  # Разумный диапазон
                            linear_dependencies.append({
                                'dimensions': (i, j),
                                'ratio': ratio,
                                'position': idx,
                                'strength': abs(grad_values[i] * grad_values[j])
                            })
        
        return gradients, linear_dependencies
    
    def spectral_collision_analysis(self, hypercube, mass_dim=5):
        """
        Спектральный анализ для обнаружения коллизий и резонансов.
        
        Параметры:
        - hypercube: гиперкуб данных
        - mass_dim: индекс параметра, соответствующего инвариантной массе
        
        Возвращает:
        - collision_map: карта коллизий
        - resonances: обнаруженные резонансы
        """
        # Проекция на ось массы
        mass_projection = np.sum(hypercube, axis=tuple(i for i in range(len(hypercube.shape)) if i != mass_dim))
        
        # Нормализация
        mass_projection = mass_projection / np.sum(mass_projection)
        
        # Вычисление DFT
        dft = fft(mass_projection)
        
        # Вычисление спектральной плотности
        spectral_density = np.abs(dft)
        
        # Поиск локальных максимумов (резонансов)
        resonances = []
        window_size = 5  # Размер окна для поиска максимумов
        
        for i in range(window_size, len(spectral_density) - window_size):
            if all(spectral_density[i] > spectral_density[i-j] for j in range(1, window_size+1)):
                # Проверяем, является ли пик статистически значимым
                background = np.mean(np.concatenate([
                    spectral_density[max(0, i-2*window_size):i-window_size],
                    spectral_density[i+window_size:min(len(spectral_density), i+2*window_size)]
                ]))
                significance = (spectral_density[i] - background) / (np.std(spectral_density) + 1e-10)
                
                if significance > 3.0:  # 3 sigma уровень
                    resonances.append({
                        'mass_bin': i,
                        'amplitude': spectral_density[i],
                        'significance': significance,
                        'frequency': i / len(spectral_density)
                    })
        
        # Карта коллизий (локальная статистическая значимость)
        collision_map = np.zeros_like(mass_projection)
        for i in range(len(mass_projection)):
            # Вычисляем локальную статистику
            window = mass_projection[max(0, i-window_size):min(len(mass_projection), i+window_size+1)]
            mean = np.mean(window)
            std = np.std(window) + 1e-10
            collision_map[i] = (mass_projection[i] - mean) / std
        
        return collision_map, resonances
    
    def multi_level_analysis(self, events, num_bins_per_dim=50):
        """
        Многослойный анализ данных на разных уровнях детализации.
        
        Параметры:
        - events: события для анализа
        - num_bins_per_dim: базовое количество ячеек по каждой координате
        
        Возвращает:
        - results: результаты анализа на всех уровнях
        """
        results = []
        
        # Анализ на разных уровнях детализации
        for level in range(1, self.num_levels + 1):
            # Построение гиперкуба с разной детализацией
            current_bins = num_bins_per_dim // (2 ** (level - 1))
            if current_bins < 3:  # Минимально допустимая детализация
                break
                
            hypercube = self.build_hypercube(events, num_bins_per_dim=current_bins)
            
            # Сжатие данных
            compressed, compression_stats = self.adaptive_compression(hypercube, level=level)
            
            # Вычисление персистентной гомологии
            diagrams = self.compute_persistent_homology(hypercube)
            betti_numbers = self.compute_betti_numbers(diagrams)
            
            # Вычисление индикатора аномалии
            anomaly_indicator, _ = self.compute_anomaly_indicator(hypercube)
            
            # Градиентный анализ
            _, linear_dependencies = self.gradient_analysis(hypercube)
            
            # Спектральный анализ коллизий
            collision_map, resonances = self.spectral_collision_analysis(hypercube)
            
            # Сохранение результатов
            results.append({
                'level': level,
                'bins_per_dim': current_bins,
                'hypercube': hypercube,
                'compressed': compressed,
                'compression_stats': compression_stats,
                'betti_numbers': betti_numbers,
                'anomaly_indicator': anomaly_indicator,
                'linear_dependencies': linear_dependencies,
                'collision_map': collision_map,
                'resonances': resonances
            })
        
        # Вычисление согласованности между уровнями
        if len(results) > 1:
            consistency = 0
            count = 0
            
            for i in range(len(results) - 1):
                for j in range(i + 1, len(results)):
                    dist = self.compute_wasserstein_distance(
                        results[i]['diagrams'][0], 
                        results[j]['diagrams'][0]
                    )
                    consistency += dist
                    count += 1
            
            if count > 0:
                consistency /= count
            results[0]['consistency'] = consistency
        
        return results
    
    def detect_new_particles(self, events, num_bins_per_dim=50, anomaly_threshold=0.5, resonance_threshold=5.0):
        """
        Обнаружение новых частиц или физических явлений в данных.
        
        Параметры:
        - events: события для анализа
        - num_bins_per_dim: количество ячеек по каждой координате
        - anomaly_threshold: порог для индикатора аномалии
        - resonance_threshold: порог статистической значимости для резонансов
        
        Возвращает:
        - discoveries: список потенциальных новых физических явлений
        """
        # Многослойный анализ
        results = self.multi_level_analysis(events, num_bins_per_dim)
        
        discoveries = []
        
        # Анализ результатов
        for result in results:
            # Проверка аномалии
            if result['anomaly_indicator'] > anomaly_threshold:
                # Проверка резонансов
                significant_resonances = [
                    r for r in result['resonances'] 
                    if r['significance'] > resonance_threshold
                ]
                
                # Проверка согласованности (если доступна)
                consistent = True
                if 'consistency' in results[0] and results[0]['consistency'] > 0.3:
                    consistent = False
                
                if significant_resonances and consistent:
                    for resonance in significant_resonances:
                        discoveries.append({
                            'type': 'potential_new_particle',
                            'mass_bin': resonance['mass_bin'],
                            'significance': resonance['significance'],
                            'anomaly_indicator': result['anomaly_indicator'],
                            'level': result['level'],
                            'betti_numbers': result['betti_numbers']
                        })
            
            # Проверка на наличие необычных топологических структур
            betti = result['betti_numbers']
            if len(betti) >= 3 and betti[0] == 1 and betti[1] >= 2 and betti[2] >= 1:
                discoveries.append({
                    'type': 'topological_anomaly',
                    'betti_numbers': betti,
                    'level': result['level'],
                    'description': 'Обнаружена топологическая структура, напоминающая тор'
                })
        
        # Сохранение обнаруженных аномалий
        self.anomalies = discoveries
        
        return discoveries
    
    def visualize_results(self, events, discoveries=None, num_bins_per_dim=50):
        """
        Визуализация результатов анализа.
        
        Параметры:
        - events: события для анализа
        - discoveries: обнаруженные аномалии (опционально)
        - num_bins_per_dim: количество ячеек по каждой координате
        """
        if discoveries is None:
            discoveries = self.anomalies
        
        # Многослойный анализ для визуализации
        results = self.multi_level_analysis(events, num_bins_per_dim)
        
        plt.figure(figsize=(20, 15))
        
        # 1. Визуализация гиперкуба (только первые два измерения для простоты)
        plt.subplot(2, 3, 1)
        if len(events[0]) >= 2:
            plt.hist2d(events[:, 0], events[:, 1], bins=50, cmap='viridis')
            plt.colorbar(label='Число событий')
            plt.xlabel(self.parameters[0])
            plt.ylabel(self.parameters[1])
            plt.title('Распределение событий')
        
        # 2. Персистентные диаграммы
        plt.subplot(2, 3, 2)
        plot_diagrams(results[0]['diagrams'], show=False)
        plt.title('Персистентные диаграммы')
        
        # 3. Карта аномалий
        plt.subplot(2, 3, 3)
        if len(results[0]['collision_map']) > 0:
            mass_bins = np.arange(len(results[0]['collision_map']))
            plt.plot(mass_bins, results[0]['collision_map'])
            plt.xlabel('Бин массы')
            plt.ylabel('Статистическая значимость')
            plt.title('Карта коллизий')
            
            # Отметим значимые резонансы
            for resonance in results[0]['resonances']:
                if resonance['significance'] > 3.0:
                    plt.axvline(x=resonance['mass_bin'], color='r', alpha=0.3)
        
        # 4. Спектр масс
        plt.subplot(2, 3, 4)
        if len(results[0]['collision_map']) > 0:
            mass_projection = np.sum(results[0]['hypercube'], 
                                    axis=tuple(i for i in range(len(results[0]['hypercube'].shape)) if i != 5))
            mass_bins = np.arange(len(mass_projection))
            plt.plot(mass_bins, mass_projection)
            plt.xlabel('Бин массы')
            plt.ylabel('Плотность событий')
            plt.title('Спектр инвариантной массы')
            
            # Отметим значимые резонансы
            for resonance in results[0]['resonances']:
                if resonance['significance'] > 3.0:
                    plt.axvline(x=resonance['mass_bin'], color='r', alpha=0.3)
        
        # 5. Индикаторы аномалий на разных уровнях
        plt.subplot(2, 3, 5)
        levels = [r['level'] for r in results]
        anomalies = [r['anomaly_indicator'] for r in results]
        plt.plot(levels, anomalies, 'o-')
        plt.xlabel('Уровень детализации')
        plt.ylabel('Индикатор аномалии')
        plt.title('Индикатор аномалии по уровням')
        plt.grid(True)
        
        # 6. Обнаруженные частицы
        plt.subplot(2, 3, 6)
        if discoveries:
            for i, discovery in enumerate(discoveries):
                if discovery['type'] == 'potential_new_particle':
                    plt.scatter(discovery['mass_bin'], discovery['significance'], 
                               s=100, c='red', marker='o')
                    plt.text(discovery['mass_bin'], discovery['significance'], 
                            f"Сигма: {discovery['significance']:.1f}", 
                            verticalalignment='bottom')
            plt.xlabel('Бин массы')
            plt.ylabel('Статистическая значимость')
            plt.title('Обнаруженные потенциальные новые частицы')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('cern_analysis_results.png', dpi=300)
        plt.show()
    
    def generate_standard_model_events(self, num_events=10000, noise_level=0.1):
        """
        Генерация синтетических данных, имитирующих стандартную модель.
        
        Параметры:
        - num_events: количество событий для генерации
        - noise_level: уровень шума
        
        Возвращает:
        - events: синтетические события
        """
        # Для примера, создадим данные, соответствующие стандартной модели
        # В реальности это должны быть данные, полученные из симуляций
        
        # Создаем события с нормальным распределением для каждого параметра
        events = np.zeros((num_events, self.n))
        
        # Псевдобыстрота (eta) - равномерное распределение
        events[:, 0] = np.random.uniform(-2.5, 2.5, num_events)  # eta1
        events[:, 3] = np.random.uniform(-2.5, 2.5, num_events)  # eta2
        
        # Азимутальный угол (phi) - равномерное распределение
        events[:, 1] = np.random.uniform(0, 2*np.pi, num_events)  # phi1
        events[:, 4] = np.random.uniform(0, 2*np.pi, num_events)  # phi2
        
        # Поперечный импульс (pT) - экспоненциальное распределение
        events[:, 2] = np.random.exponential(50, num_events)
        
        # Инвариантная масса - распределение, имитирующее стандартную модель
        # Например, комбинация фоновых процессов и известных резонансов
        background = np.random.exponential(100, num_events)
        z_boson = np.random.normal(91, 5, num_events//5)  # Z-бозон
        higgs = np.random.normal(125, 10, num_events//20)  # Хиггс
        
        mass = np.concatenate([background, z_boson, higgs])
        np.random.shuffle(mass)
        events[:, 5] = mass[:num_events]
        
        # Добавляем шум
        events += np.random.normal(0, noise_level, events.shape)
        
        return events
    
    def generate_crypto_events(self, num_events=10000, d=27, noise_level=0.05):
        """
        Генерация синтетических данных, имитирующих крипто-подобные события.
        
        Параметры:
        - num_events: количество событий для генерации
        - d: "приватный ключ" для создания структуры
        - noise_level: уровень шума
        
        Возвращает:
        - events: синтетические крипто-подобные события
        """
        events = np.zeros((num_events, self.n))
        
        # Генерация событий с крипто-подобной структурой
        for i in range(num_events):
            # Генерируем случайные параметры
            u_r = np.random.uniform(0, 1)
            u_z = np.random.uniform(0, 1)
            
            # Добавляем структуру, аналогичную ECDSA
            k = (u_z + u_r * d) % 1
            
            # Преобразуем в физические параметры
            events[i, 0] = u_r * 5 - 2.5  # eta1
            events[i, 1] = u_z * 2 * np.pi  # phi1
            events[i, 2] = k * 100  # pT
            events[i, 3] = (u_r * 0.8 + 0.2) * 5 - 2.5  # eta2
            events[i, 4] = (u_z * 0.9 + 0.1) * 2 * np.pi  # phi2
            events[i, 5] = k * 150  # mass
        
        # Добавляем шум
        events += np.random.normal(0, noise_level, events.shape)
        
        return events
    
    def benchmark(self, num_events_list=[1000, 5000, 10000, 20000]):
        """
        Бенчмарк производительности фреймворка.
        
        Параметры:
        - num_events_list: список количества событий для тестирования
        
        Возвращает:
        - results: результаты бенчмарка
        """
        results = []
        
        for num_events in num_events_list:
            print(f"Тестирование с {num_events} событиями...")
            
            # Генерация данных
            start_time = time.time()
            events = self.generate_standard_model_events(num_events)
            gen_time = time.time() - start_time
            
            # Построение гиперкуба
            start_time = time.time()
            hypercube = self.build_hypercube(events, num_bins_per_dim=30)
            build_time = time.time() - start_time
            
            # Сжатие данных
            start_time = time.time()
            compressed, compression_stats = self.adaptive_compression(hypercube)
            compress_time = time.time() - start_time
            
            # Многослойный анализ
            start_time = time.time()
            discoveries = self.detect_new_particles(events)
            analysis_time = time.time() - start_time
            
            results.append({
                'num_events': num_events,
                'generation_time': gen_time,
                'build_time': build_time,
                'compress_time': compress_time,
                'analysis_time': analysis_time,
                'total_time': gen_time + build_time + compress_time + analysis_time,
                'compression_ratio': compression_stats['compression_ratio'],
                'num_discoveries': len(discoveries)
            })
            
            print(f"  Время генерации: {gen_time:.4f} с")
            print(f"  Время построения гиперкуба: {build_time:.4f} с")
            print(f"  Время сжатия: {compress_time:.4f} с")
            print(f"  Время анализа: {analysis_time:.4f} с")
            print(f"  Коэффициент сжатия: {compression_stats['compression_ratio']:.4f}")
            print(f"  Обнаружено аномалий: {len(discoveries)}")
        
        # Визуализация результатов бенчмарка
        plt.figure(figsize=(12, 8))
        
        # Время выполнения
        plt.subplot(2, 1, 1)
        plt.plot(num_events_list, [r['total_time'] for r in results], 'o-')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Количество событий')
        plt.ylabel('Время выполнения (с)')
        plt.title('Производительность QCPH')
        plt.grid(True, which="both", ls="-")
        
        # Коэффициент сжатия
        plt.subplot(2, 1, 2)
        plt.plot(num_events_list, [r['compression_ratio'] for r in results], 'o-')
        plt.xscale('log')
        plt.xlabel('Количество событий')
        plt.ylabel('Коэффициент сжатия')
        plt.grid(True, which="both", ls="-")
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300)
        plt.show()
        
        return results

def demo():
    """
    Демонстрация работы Quantum-Crypto-Particle Hybrid Framework.
    """
    print("="*80)
    print("Quantum-Crypto-Particle Hybrid Framework (QCPH) v1.0")
    print("Демонстрация работы системы для обнаружения новых физических явлений в данных CERN")
    print("="*80)
    
    # Инициализация фреймворка
    parameters = ['eta1', 'phi1', 'pT', 'eta2', 'phi2', 'mass']
    qcph = CERNHypercubeFramework(parameters=parameters)
    
    # Генерация синтетических данных
    print("\nГенерация синтетических данных...")
    standard_events = qcph.generate_standard_model_events(num_events=15000)
    crypto_events = qcph.generate_crypto_events(num_events=5000, d=27)
    
    # Объединение данных (крипто-события как аномалии в стандартных данных)
    all_events = np.vstack([standard_events, crypto_events])
    
    # Установка стандартной модели
    print("\nУстановка эталонной модели...")
    qcph.set_standard_model(standard_events)
    
    # Анализ данных
    print("\nАнализ данных для обнаружения аномалий...")
    discoveries = qcph.detect_new_particles(all_events)
    
    # Вывод результатов
    print("\nРЕЗУЛЬТАТЫ АНАЛИЗА:")
    if discoveries:
        print(f"Обнаружено {len(discoveries)} потенциальных новых физических явлений:")
        for i, discovery in enumerate(discoveries):
            if discovery['type'] == 'potential_new_particle':
                print(f"  {i+1}. Потенциальная новая частица (масса={discovery['mass_bin']}, значимость={discovery['significance']:.2f}σ)")
                print(f"     Индикатор аномалии: {discovery['anomaly_indicator']:.4f}")
                print(f"     Топологические инварианты: β = {discovery['betti_numbers']}")
            elif discovery['type'] == 'topological_anomaly':
                print(f"  {i+1}. Топологическая аномалия")
                print(f"     Топологические инварианты: β = {discovery['betti_numbers']}")
                print(f"     Описание: {discovery['description']}")
    else:
        print("  Аномалий не обнаружено")
    
    # Визуализация результатов
    print("\nГенерация визуализации результатов...")
    qcph.visualize_results(all_events, discoveries)
    
    # Бенчмарк производительности
    print("\nЗапуск бенчмарка производительности...")
    benchmark_results = qcph.benchmark()
    
    print("\nДемонстрация завершена.")
    print("Результаты сохранены в файлы: cern_analysis_results.png, benchmark_results.png")
    print("="*80)

if __name__ == "__main__":
    demo()
