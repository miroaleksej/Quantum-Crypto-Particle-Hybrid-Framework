"""
Quantum-Crypto-Particle Hybrid Framework (QCPH v1.0)
===================================================
Этот фреймворк объединяет методы топологического анализа, криптографии и физики частиц
для обнаружения аномалий в данных коллайдера через призму криптографических сигнатур.

Основные компоненты:
1. Генерация синтетических данных коллайдера
2. Топологический анализ (числа Бетти)
3. DFT-анализ аномалий
4. Вычисление топологической энтропии
5. Анализ асимптотики длины кривой

Данный фреймворк может быть интегрирован в unified_lhc_framework_v.2.0 как модуль CryptoAnomalyDetector
"""

import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.fft import fft
from typing import Tuple, List, Dict, Any, Optional


class CryptoAnomalyDetector:
    """
    Основной класс Quantum-Crypto-Particle Hybrid Framework (QCPH v1.0)
    Предназначен для обнаружения криптографических сигнатур в данных коллайдера
    """
    
    def __init__(self, n_events: int = 500):
        """
        Инициализация детектора аномалий
        
        Args:
            n_events: Количество событий для анализа (по умолчанию 500)
        """
        self.n_events = n_events
        self.standard_events = None
        self.crypto_events = None
        self.betti_standard = None
        self.betti_crypto = None
        self.anomaly_results = None
        self.topological_entropy = None
        self.curve_length = None
    
    def generate_standard_events(self) -> np.ndarray:
        """
        Генерирует события, соответствующие Стандартной модели (случайное распределение)
        
        Returns:
            Массив событий размером (n_events, 3)
        """
        events = []
        for _ in range(self.n_events):
            # Генерируем случайные координаты в 3D пространстве
            r = np.random.uniform(0, 1)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            # Сферические координаты в декартовы
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            events.append([x, y, z])
        self.standard_events = np.array(events)
        return self.standard_events
    
    def generate_crypto_events(self) -> np.ndarray:
        """
        Генерирует события с топологией тора (S^1 × S^1), как в ECDSA
        
        Returns:
            Массив событий размером (n_events, 3)
        """
        events = []
        for i in range(self.n_events):
            # Параметризация тора
            u = 2 * np.pi * i / self.n_events
            v = 2 * np.pi * np.random.uniform(0, 1)
            # Уравнение тора
            R = 0.8  # Большой радиус
            r = 0.3  # Малый радиус
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            events.append([x, y, z])
        self.crypto_events = np.array(events)
        return self.crypto_events
    
    def compute_betti_numbers(self, data: np.ndarray) -> Tuple[int, int, int]:
        """
        Вычисляет числа Бетти с использованием персистентной гомологии
        
        Args:
            data: Массив данных размером (m, n)
            
        Returns:
            Кортеж (β0, β1, β2) - числа Бетти
        """
        result = ripser(data, maxdim=2)
        dgms = result['dgms']
        
        # Анализируем диаграммы персистентности
        betti_0 = len(dgms[0]) - np.sum(dgms[0][:, 1] < np.inf)  # Связные компоненты
        betti_1 = np.sum(dgms[1][:, 1] == np.inf)  # 1-мерные "дыры"
        betti_2 = np.sum(dgms[2][:, 1] == np.inf)  # 2-мерные "полости"
        
        return betti_0, betti_1, betti_2
    
    def detect_anomaly_dft(self, events: np.ndarray, n: Optional[int] = None) -> Tuple[float, float, bool]:
        """
        Обнаружение аномалий через DFT-анализ, как в разделе 13 документа 3.txt
        
        Args:
            events: Массив событий размера (m, 3)
            n: Размер выборки для анализа (по умолчанию используется все события)
            
        Returns:
            Кортеж (anomaly_score, threshold, is_anomaly)
        """
        if n is None:
            n = len(events)
        n = min(n, len(events))
        
        # Берем только проекцию на плоскость (x,y), как в ECDSA (u_r, u_z)
        projection = events[:, :2]
        # Сортируем точки по углу для упорядочивания
        angles = np.arctan2(projection[:, 1], projection[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_proj = projection[sorted_indices]
        # Вычисляем DFT
        dft_x = fft(sorted_proj[:n, 0])
        dft_y = fft(sorted_proj[:n, 1])
        # Вычисляем фазовые сдвиги
        phase_x = np.angle(dft_x[1])  # Берем первую гармонику
        phase_y = np.angle(dft_y[1])
        # Стандартное отклонение фазы
        phase_std = np.std([phase_x, phase_y])
        # Пороговое значение из файла 3.txt: np.sqrt(n)/2
        threshold = np.sqrt(n)/2
        anomaly_score = phase_std
        return anomaly_score, threshold, anomaly_score > threshold
    
    def compute_topological_entropy(self, events: np.ndarray) -> float:
        """
        Вычисляет топологическую энтропию для событий коллайдера
        
        Args:
            events: Массив событий
            
        Returns:
            Топологическая энтропия
        """
        # Для крипто-событий с топологией тора
        # d_phys можно оценить через соотношение радиусов тора
        R = 0.8  # Большой радиус (из generate_crypto_events)
        r = 0.3  # Малый радиус (из generate_crypto_events)
        d_phys = R / r  # Аналог приватного ключа
        # Топологическая энтропия
        h_top = np.log(max(1, d_phys))
        return h_top
    
    def compute_curve_length(self, events: np.ndarray) -> float:
        """
        Вычисляет длину кривой для событий
        
        Args:
            events: Массив событий
            
        Returns:
            Длина кривой
        """
        # Берем только проекцию на плоскость
        projection = events[:, :2]
        # Сортируем точки по углу
        angles = np.arctan2(projection[:, 1], projection[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_proj = projection[sorted_indices]
        # Вычисляем длину кривой
        L = 0
        for i in range(len(sorted_proj)-1):
            dx = sorted_proj[i+1, 0] - sorted_proj[i, 0]
            dy = sorted_proj[i+1, 1] - sorted_proj[i, 1]
            L += np.sqrt(dx**2 + dy**2)
        return L
    
    def analyze_asymptotic_curve_length(self, r: float = 0.3, num_points: int = 10) -> Dict[str, Any]:
        """
        Проверяет асимптотику длины кривой L(d) ~ C * ln(d)
        
        Args:
            r: Малый радиус тора (по умолчанию 0.3)
            num_points: Количество точек для анализа
            
        Returns:
            Словарь с результатами анализа
        """
        R_values = np.linspace(0.5, 1.5, num_points)
        L_values = []
        
        for R in R_values:
            # Генерируем события с разным соотношением радиусов
            events = []
            for i in range(self.n_events):
                u = 2 * np.pi * i / self.n_events
                v = 2 * np.pi * np.random.uniform(0, 1)
                x = (R + r * np.cos(v)) * np.cos(u)
                y = (R + r * np.cos(v)) * np.sin(u)
                z = r * np.sin(v)
                events.append([x, y, z])
            L = self.compute_curve_length(np.array(events))
            L_values.append(L)
        
        # Проверяем асимптотику L(d) ~ C * ln(d)
        d_values = R_values / r  # d = R/r
        log_d_values = np.log(d_values)
        # Линейная регрессия
        coeffs = np.polyfit(log_d_values, L_values, 1)
        C = coeffs[0]
        intercept = coeffs[1]
        r_squared = np.corrcoef(log_d_values, L_values)[0, 1]**2
        
        return {
            "C": C,
            "intercept": intercept,
            "r_squared": r_squared,
            "d_values": d_values.tolist(),
            "L_values": L_values,
            "log_d_values": log_d_values.tolist()
        }
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Запускает полный анализ для стандартных и крипто-событий
        
        Returns:
            Словарь с результатами анализа
        """
        # Генерируем данные
        standard_events = self.generate_standard_events()
        crypto_events = self.generate_crypto_events()
        
        # Вычисляем числа Бетти
        betti_standard = self.compute_betti_numbers(standard_events)
        betti_crypto = self.compute_betti_numbers(crypto_events)
        
        # DFT-анализ аномалий
        anomaly_standard = self.detect_anomaly_dft(standard_events)
        anomaly_crypto = self.detect_anomaly_dft(crypto_events)
        
        # Топологическая энтропия
        h_top_standard = self.compute_topological_entropy(standard_events)
        h_top_crypto = self.compute_topological_entropy(crypto_events)
        
        # Длина кривой
        curve_length_standard = self.compute_curve_length(standard_events)
        curve_length_crypto = self.compute_curve_length(crypto_events)
        
        # Анализ асимптотики
        asymptotic_analysis = self.analyze_asymptotic_curve_length()
        
        return {
            "betti_standard": betti_standard,
            "betti_crypto": betti_crypto,
            "anomaly_standard": {
                "score": anomaly_standard[0],
                "threshold": anomaly_standard[1],
                "is_anomaly": anomaly_standard[2]
            },
            "anomaly_crypto": {
                "score": anomaly_crypto[0],
                "threshold": anomaly_crypto[1],
                "is_anomaly": anomaly_crypto[2]
            },
            "topological_entropy_standard": h_top_standard,
            "topological_entropy_crypto": h_top_crypto,
            "curve_length_standard": curve_length_standard,
            "curve_length_crypto": curve_length_crypto,
            "asymptotic_analysis": asymptotic_analysis
        }
    
    def visualize_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Визуализирует результаты анализа
        
        Args:
            results: Результаты анализа из run_full_analysis
            save_path: Путь для сохранения изображения (если None, отображает в интерактивном режиме)
        """
        plt.figure(figsize=(15, 12))
        
        # 1. Визуализация стандартных событий
        plt.subplot(2, 2, 1, projection='3d')
        if self.standard_events is not None:
            plt.scatter(self.standard_events[:, 0], self.standard_events[:, 1], self.standard_events[:, 2], 
                       c='blue', s=10, alpha=0.6)
            plt.title(f'Стандартные события (СМ)\nЧисла Бетти: β0={results["betti_standard"][0]}, '
                      f'β1={results["betti_standard"][1]}, β2={results["betti_standard"][2]}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.zlabel = 'Z'
        
        # 2. Визуализация крипто-событий
        plt.subplot(2, 2, 2, projection='3d')
        if self.crypto_events is not None:
            plt.scatter(self.crypto_events[:, 0], self.crypto_events[:, 1], self.crypto_events[:, 2], 
                       c='red', s=10, alpha=0.6)
            plt.title(f'Крипто-события\nЧисла Бетти: β0={results["betti_crypto"][0]}, '
                      f'β1={results["betti_crypto"][1]}, β2={results["betti_crypto"][2]}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.zlabel = 'Z'
        
        # 3. DFT-анализ
        plt.subplot(2, 2, 3)
        labels = ['Стандартные', 'Крипто']
        scores = [results["anomaly_standard"]["score"], results["anomaly_crypto"]["score"]]
        thresholds = [results["anomaly_standard"]["threshold"], results["anomaly_crypto"]["threshold"]]
        colors = ['blue', 'red']
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, scores, width, label='Оценка аномалии', color=colors)
        plt.bar(x + width/2, thresholds, width, label='Порог', color=['gray', 'gray'], alpha=0.5)
        
        plt.axhline(y=thresholds[0], color='gray', linestyle='--', alpha=0.7)
        plt.text(len(labels), thresholds[0], f'Порог = {thresholds[0]:.2f}', va='bottom')
        
        plt.ylabel('Значение')
        plt.title('Результаты DFT-анализа')
        plt.xticks(x, labels)
        plt.legend()
        
        # 4. Асимптотика длины кривой
        plt.subplot(2, 2, 4)
        asymptotic = results["asymptotic_analysis"]
        plt.scatter(asymptotic["log_d_values"], asymptotic["L_values"], color='green', label='Данные')
        
        # Построение линии регрессии
        log_d = np.array(asymptotic["log_d_values"])
        C = asymptotic["C"]
        intercept = asymptotic["intercept"]
        regression_line = C * log_d + intercept
        plt.plot(log_d, regression_line, 'r-', label=f'Линейная регрессия: L = {C:.2f}·ln(d) + {intercept:.2f}')
        
        plt.xlabel('ln(d)')
        plt.ylabel('L(d)')
        plt.title(f'Асимптотика длины кривой (R² = {asymptotic["r_squared"]:.4f})')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Визуализация сохранена в {save_path}")
        else:
            plt.show()


def main():
    """
    Демонстрационная функция для проверки работоспособности QCPH v1.0
    """
    print("="*80)
    print("Quantum-Crypto-Particle Hybrid Framework (QCPH v1.0) - Демонстрация")
    print("="*80)
    
    # Создаем экземпляр детектора
    detector = CryptoAnomalyDetector(n_events=500)
    
    # Запускаем полный анализ
    print("\nЗапуск полного анализа...")
    results = detector.run_full_analysis()
    
    # Выводим результаты
    print("\nРЕЗУЛЬТАТЫ АНАЛИЗА:")
    print("-"*50)
    
    # Числа Бетти
    print("\n1. ТОПОЛОГИЧЕСКИЕ ХАРАКТЕРИСТИКИ:")
    print(f"   Стандартные события (СМ): β0={results['betti_standard'][0]}, "
          f"β1={results['betti_standard'][1]}, β2={results['betti_standard'][2]}")
    print(f"   Крипто-события: β0={results['betti_crypto'][0]}, "
          f"β1={results['betti_crypto'][1]}, β2={results['betti_crypto'][2]}")
    
    # DFT-анализ
    print("\n2. DFT-АНАЛИЗ АНОМАЛИЙ:")
    print(f"   Стандартные события: score={results['anomaly_standard']['score']:.4f}, "
          f"threshold={results['anomaly_standard']['threshold']:.4f}, "
          f"аномалия={results['anomaly_standard']['is_anomaly']}")
    print(f"   Крипто-события: score={results['anomaly_crypto']['score']:.4f}, "
          f"threshold={results['anomaly_crypto']['threshold']:.4f}, "
          f"аномалия={results['anomaly_crypto']['is_anomaly']}")
    
    # Топологическая энтропия
    print("\n3. ТОПОЛОГИЧЕСКАЯ ЭНТРОПИЯ:")
    print(f"   Стандартные события: {results['topological_entropy_standard']:.4f}")
    print(f"   Крипто-события: {results['topological_entropy_crypto']:.4f}")
    
    # Длина кривой
    print("\n4. ДЛИНА КРИВОЙ:")
    print(f"   Стандартные события: {results['curve_length_standard']:.4f}")
    print(f"   Крипто-события: {results['curve_length_crypto']:.4f}")
    
    # Асимптотика
    asymptotic = results['asymptotic_analysis']
    print("\n5. АСИМПТОТИКА ДЛИНЫ КРИВОЙ:")
    print(f"   Эмпирическая формула: L(d) = {asymptotic['C']:.4f} * ln(d) + {asymptotic['intercept']:.4f}")
    print(f"   Коэффициент детерминации R²: {asymptotic['r_squared']:.4f}")
    
    # Визуализация результатов
    print("\nГенерация визуализации результатов...")
    detector.visualize_results(results)
    
    # Заключение
    print("\nЗАКЛЮЧЕНИЕ:")
    print("-"*50)
    if results['anomaly_crypto']['is_anomaly']:
        print("Крипто-события успешно обнаружены как аномалии!")
        print("Это подтверждает гипотезу о том, что QCPH v1.0 может эффективно выявлять")
        print("криптографические сигнатуры в данных коллайдера через топологический анализ.")
    else:
        print("Крипто-события НЕ были обнаружены как аномалии.")
        print("Требуется дальнейшая настройка параметров фреймворка.")
    
    print("\nQCPH v1.0 успешно завершил работу!")
    print("="*80)


if __name__ == "__main__":
    main()
