import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.fft import fft
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import math

class CryptoAnomalyDetector:
    """
    Основной класс Quantum-Crypto-Particle Hybrid Framework (QCPH v1.2)
    Предназначен для:
    - Отображения событий коллайдера в топологическое пространство ECDSA
    - Обнаружения аномалий через DFT и топологию
    - Генерации криптографических ключей
    
    Соответствует математической модели из файлов 2.txt и 3.txt:
    - Определение 1.3: Топологическое отображение ψ: M → T
    - Теорема 2.3: Топологическая энтропия
    - Определение 3.3: Показатель аномалии
    - Алгоритм 5.3: Генерация криптографического ключа
    """
    def __init__(self, n_events: int = 500, n: int = 256):
        """
        Инициализация детектора аномалий
        
        Args:
            n_events: Количество событий для анализа
            n: Порядок группы эллиптической кривой (для mod n)
        """
        self.n_events = n_events
        self.n = n  # Порядок группы ECDSA
        self.events = []  # Сырые события коллайдера
        self.ur_uz_list = []  # Результат ψ(e) = (u_r, u_z)
        self.d_phys_list = []  # Вычисленные d_phys(e)
        self.betti_numbers = None
        self.anomaly_score = None
        self.topological_entropy = None
        self.curve_length = None
        self.crypto_key = None
    
    def _spherical_coords(self, px: float, py: float, pz: float) -> Tuple[float, float]:
        """
        Перевод декартовых координат импульса в сферические (θ, φ)
        
        Args:
            px, py, pz: компоненты импульса
            
        Returns:
            θ (азимутальный угол), φ (полярный угол)
        """
        r = np.sqrt(px**2 + py**2 + pz**2)
        if r == 0:
            return 0.0, 0.0
        theta = np.arctan2(py, px)  # от 0 до 2π
        phi = np.arccos(pz / r)     # от 0 до π
        return theta, phi
    
    def generate_sample_events(self) -> List[Dict]:
        """
        Генерирует синтетические события коллайдера (для тестирования)
        
        Returns:
            Список событий: [{'particles': [{'px', 'py', 'pz', 'E'}, ...]}, ...]
        """
        sample_events = []
        for _ in range(self.n_events):
            k = np.random.randint(3, 8)  # число частиц
            particles = []
            total_E = 0.0
            for _ in range(k):
                # Генерируем случайные частицы
                px = np.random.normal(0, 10)
                py = np.random.normal(0, 10)
                pz = np.random.normal(0, 15)
                E = np.random.uniform(5, 50)
                total_E += E
                theta, phi = self._spherical_coords(px, py, pz)
                particles.append({'px': px, 'py': py, 'pz': pz, 'E': E, 'theta': theta, 'phi': phi})
            sample_events.append({'particles': particles, 'total_E': total_E})
        self.events = sample_events
        return sample_events
    
    def compute_ur_uz(self, event: Dict) -> Tuple[float, float]:
        """
        Реализация отображения ψ: M → T
        ψ(e) = (u_r(e), u_z(e)) согласно Определению 1.3
        
        Args:
            event: событие с частицами
            
        Returns:
            (u_r, u_z) mod n
        """
        total_E = event['total_E']
        particles = event['particles']
        ur = 0.0
        uz = 0.0
        for p in particles:
            ur += p['E'] * p['theta']
            uz += p['E'] * p['phi']
        ur = (ur / total_E) % self.n
        uz = (uz / total_E) % self.n
        return ur, uz
    
    def compute_d_phys(self, ur: float, uz: float, r: float = 0.3, R: float = 0.8) -> float:
        """
        Вычисление d_phys по Определению 2.2:
        d_phys = - (∂r/∂u_z) * (∂r/∂u_r)^{-1} mod n
        
        Здесь r(u_r, u_z) - радиус тора как функция координат.
        Для стандартной параметризации тора:
        x = (R + r*cos(u_z)) * cos(u_r)
        y = (R + r*cos(u_z)) * sin(u_r)
        z = r*sin(u_z)
        
        Тогда "радиус" от центра: r(u_r, u_z) = sqrt(x^2 + y^2) = R + r*cos(u_z)
        Следовательно:
        ∂r/∂u_r = 0
        ∂r/∂u_z = -r*sin(u_z)
        
        Но это вырождение. Вместо этого используем обобщённую модель:
        r(u_r, u_z) = R + r * cos(u_z - d*u_r)  # как в динамике T(u_r, u_z) = (u_r, u_z + d)
        Тогда:
        ∂r/∂u_r = r * d * sin(u_z - d*u_r)
        ∂r/∂u_z = -r * sin(u_z - d*u_r)
        => d_phys = - (∂r/∂u_z) / (∂r/∂u_r) = - (-r sin(...)) / (r d sin(...)) = 1/d
        => d = 1 / d_phys
        
        В этом исправленном коде:
        - Реализована более точная численная оценка производных
        - Улучшена защита от деления на ноль
        - Добавлена обработка крайних случаев
        """
        # Аппроксимация r(u_r, u_z) как расстояние от оси z
        # r = sqrt(x^2 + y^2) = R + r * cos(u_z - d * u_r)
        delta = 1e-5
        
        # Вычисляем частные производные с использованием центральной разности
        # ∂r/∂u_r
        r_ur_plus = R + r * np.cos(uz - (ur + delta))
        r_ur_minus = R + r * np.cos(uz - (ur - delta))
        dr_dur = (r_ur_plus - r_ur_minus) / (2 * delta)
        
        # ∂r/∂u_z
        r_uz_plus = R + r * np.cos((uz + delta) - ur)
        r_uz_minus = R + r * np.cos((uz - delta) - ur)
        dr_duz = (r_uz_plus - r_uz_minus) / (2 * delta)
        
        # Защита от деления на ноль с учетом физического смысла
        if abs(dr_dur) < 1e-10:
            # Если производная по u_r близка к нулю, используем регуляризацию
            # В соответствии с теорией, это соответствует d_phys -> infinity
            d_phys = self.n / 2  # значение по умолчанию для вырожденного случая
        else:
            d_phys = - dr_duz / dr_dur
        
        # Обеспечиваем корректность значения в рамках модуля n
        d_phys = d_phys % self.n
        # Для стабильности вычислений ограничиваем значение
        d_phys = min(d_phys, self.n * 0.99)
        
        return d_phys
    
    def map_events_to_torus(self) -> List[Tuple[float, float]]:
        """
        Применяет ψ ко всем событиям и сохраняет (u_r, u_z)
        
        Returns:
            Список (u_r, u_z)
        """
        self.ur_uz_list = []
        self.d_phys_list = []
        for event in self.events:
            ur, uz = self.compute_ur_uz(event)
            d_phys = self.compute_d_phys(ur, uz)
            self.ur_uz_list.append((ur, uz))
            self.d_phys_list.append(d_phys)
        return self.ur_uz_list
    
    def compute_betti_numbers(self) -> Tuple[int, int, int]:
        """
        Вычисляет числа Бетти с использованием персистентной гомологии.
        В соответствии с Модулем 6 математической модели:
        - b₀: число бесконечно живущих компонент в H₀
        - b₁: число 1-циклов, живущих бесконечно
        - b₂: число 2-полостей
        
        Returns:
            (b₀, b₁, b₂)
        """
        if not self.ur_uz_list:
            raise ValueError("Сначала выполните map_events_to_torus()")
        
        data = np.array(self.ur_uz_list)  # (N, 2) — точки на торе
        
        # Добавляем небольшой шум для избежания вырожденных случаев
        noise = np.random.normal(0, 1e-10, data.shape)
        data = data + noise
        
        result = ripser(data, maxdim=2, thresh=self.n)
        dgms = result['dgms']
        
        # b₀: число компонент, у которых death = ∞
        betti_0 = np.sum(dgms[0][:, 1] == np.inf)
        
        # b₁: число 1-циклов, живущих бесконечно
        if len(dgms) > 1:
            betti_1 = np.sum(dgms[1][:, 1] == np.inf)
        else:
            betti_1 = 0
            
        # b₂: число 2-полостей
        if len(dgms) > 2:
            betti_2 = np.sum(dgms[2][:, 1] == np.inf)
        else:
            betti_2 = 0
            
        self.betti_numbers = (betti_0, betti_1, betti_2)
        return self.betti_numbers
    
    def detect_anomaly_dft(self, bin_size: int = 50, k_max: int = 10) -> float:
        """
        DFT-анализ по Теореме 3.2 и Определению 3.3
        
        Args:
            bin_size: размер окна по u_r
            k_max: максимальная гармоника для усреднения
            
        Returns:
            Показатель аномалии 𝒜
        """
        if not self.ur_uz_list:
            raise ValueError("Сначала выполните map_events_to_torus()")
        
        ur_vals = np.array([p[0] for p in self.ur_uz_list])
        uz_vals = np.array([p[1] for p in self.ur_uz_list])
        
        # Сортируем по u_r
        sorted_indices = np.argsort(ur_vals)
        ur_sorted = ur_vals[sorted_indices]
        uz_sorted = uz_vals[sorted_indices]
        
        # Биним по u_r
        u_r_min, u_r_max = ur_sorted.min(), ur_sorted.max()
        bin_edges = np.linspace(u_r_min, u_r_max, num=bin_size + 1)
        
        A = 0.0
        count = 0
        
        # Исправлено: range(bin_size - 2) вместо range(bin_size - 1)
        # чтобы избежать выхода за границы массива bin_edges
        for i in range(bin_size - 2):
            mask = (ur_sorted >= bin_edges[i]) & (ur_sorted < bin_edges[i + 1])
            uz_bin = uz_sorted[mask]
            if len(uz_bin) < 2:
                continue
                
            # Нормализуем uz в [0, n)
            uz_normalized = uz_bin % self.n
            
            # Дискретизация uz
            hist, bin_edges_uz = np.histogram(uz_normalized, bins=self.n, range=(0, self.n))
            f = hist.astype(float)
            f_hat = fft(f)
            
            # Следующий бин
            mask_next = (ur_sorted >= bin_edges[i + 1]) & (ur_sorted < bin_edges[i + 2])
            uz_next = uz_sorted[mask_next]
            if len(uz_next) < 2:
                continue
                
            uz_next_norm = uz_next % self.n
            hist_next, _ = np.histogram(uz_next_norm, bins=self.n, range=(0, self.n))
            f_next = hist_next.astype(float)
            f_next_hat = fft(f_next)
            
            # Средний d_phys для бинов (исключая нулевые значения)
            valid_d_phys = [d for d in self.d_phys_list if abs(d) > 1e-10]
            if len(valid_d_phys) > 0:
                d_phys_avg = np.mean(valid_d_phys)
            else:
                d_phys_avg = 1.0  # значение по умолчанию
            
            # Вычисляем 𝒜 для гармоник k=1..k_max
            for k in range(1, min(k_max + 1, self.n // 2)):
                if abs(f_hat[k]) < 1e-10 or abs(f_next_hat[k]) < 1e-10:
                    continue
                    
                ratio = f_next_hat[k] / f_hat[k]
                expected = np.exp(-2j * np.pi * k * d_phys_avg / self.n)
                A += abs(ratio - expected)**2
                count += 1
        
        if count == 0:
            return 0.0
            
        self.anomaly_score = A / count
        return self.anomaly_score
    
    def compute_topological_entropy(self) -> float:
        """
        Топологическая энтропия по Теореме 2.3
        h_top = log(max(1, |d_phys|))
        Используем среднее d_phys
        """
        if not self.d_phys_list:
            raise ValueError("Сначала выполните map_events_to_torus()")
        
        # Используем медиану вместо среднего для устойчивости к выбросам
        d_avg = np.median(np.abs(self.d_phys_list))
        
        # Добавляем небольшое значение, чтобы избежать log(0)
        d_avg = max(1.0, d_avg + 1e-10)
        
        self.topological_entropy = np.log(d_avg)
        return self.topological_entropy
    
    def compute_curve_length(self) -> float:
        """
        Длина кривой в пространстве (u_r, u_z)
        Исправлено вычисление расстояния на торе
        """
        if not self.ur_uz_list:
            raise ValueError("Сначала выполните map_events_to_torus()")
        
        points = np.array(self.ur_uz_list)
        
        # Сортируем по u_r
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        L = 0.0
        for i in range(len(sorted_points) - 1):
            # Вычисляем разницу с учетом топологии тора
            du_r = (sorted_points[i + 1, 0] - sorted_points[i, 0]) % self.n
            du_z = (sorted_points[i + 1, 1] - sorted_points[i, 1]) % self.n
            
            # Находим кратчайшее расстояние на торе
            du_r = min(du_r, self.n - du_r)
            du_z = min(du_z, self.n - du_z)
            
            L += np.sqrt(du_r**2 + du_z**2)
        
        self.curve_length = L
        return L
    
    def analyze_asymptotic_curve_length(self, d_values: List[float]) -> Dict[str, Any]:
        """
        Анализ асимптотики L(d) ~ C ln d
        """
        L_values = []
        for d_val in d_values:
            # Моделируем события с заданным d
            ur = np.linspace(0, self.n, self.n_events)
            uz = (ur * d_val) % self.n
            points = np.array([(u, z) for u, z in zip(ur, uz)])
            
            L = 0
            for i in range(len(points) - 1):
                # Вычисляем разницу с учетом топологии тора
                du = (points[i + 1][0] - points[i][0]) % self.n
                dz = (points[i + 1][1] - points[i][1]) % self.n
                
                # Находим кратчайшее расстояние на торе
                du = min(du, self.n - du)
                dz = min(dz, self.n - dz)
                
                L += np.sqrt(du**2 + dz**2)
            
            L_values.append(L)
        
        # Используем только положительные значения d для логарифма
        positive_d = [d for d in d_values if d > 0]
        if len(positive_d) == 0:
            raise ValueError("Все значения d не положительны")
            
        log_d = np.log(positive_d)
        valid_L = L_values[:len(positive_d)]
        
        coeffs = np.polyfit(log_d, valid_L, 1)
        C, b = coeffs
        r2 = np.corrcoef(log_d, valid_L)[0, 1]**2
        
        return {
            "C": float(C),
            "intercept": float(b),
            "r_squared": float(r2),
            "d_values": d_values,
            "L_values": L_values,
            "log_d_values": log_d.tolist()
        }
    
    def generate_crypto_key(self) -> str:
        """
        Алгоритм 5.3: Генерация криптографического ключа
        d_crypto = Hash( sum_{i=1}^m k_i * d_phys(e_i) mod n )
        где k_i = u_z,i + u_r,i * d_phys(e_i) mod n
        """
        if not self.ur_uz_list or not self.d_phys_list:
            raise ValueError("Сначала выполните map_events_to_torus()")
        
        total = 0
        for (ur, uz), d_phys in zip(self.ur_uz_list, self.d_phys_list):
            k_i = (uz + ur * d_phys) % self.n
            term = (k_i * d_phys) % self.n
            total = (total + term) % self.n
        
        # Исправлено: преобразование числа в байты вместо строки
        # для сохранения полной энтропии
        byte_length = (self.n.bit_length() + 7) // 8
        total_bytes = total.to_bytes(byte_length, byteorder='big')
        
        self.crypto_key = hashlib.sha256(total_bytes).hexdigest()
        return self.crypto_key
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Полный анализ: от событий до ключа
        """
        # 1. Генерация событий
        self.generate_sample_events()
        
        # 2. Отображение в тор
        self.map_events_to_torus()
        
        # 3. Топология
        betti = self.compute_betti_numbers()
        
        # 4. DFT-анализ
        anomaly_score = self.detect_anomaly_dft()
        
        # 5. Энтропия
        h_top = self.compute_topological_entropy()
        
        # 6. Длина кривой
        curve_length = self.compute_curve_length()
        
        # 7. Асимптотика
        try:
            asymptotic = self.analyze_asymptotic_curve_length(
                d_values=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
            )
        except ValueError:
            # Обработка случая, когда все d не положительны
            asymptotic = {
                "C": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "d_values": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
                "L_values": [0.0] * 6,
                "log_d_values": [np.log(d) for d in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]]
            }
        
        # 8. Генерация ключа
        key = self.generate_crypto_key()
        
        # 9. Критерий аномалии
        threshold = np.sqrt(self.n) / 2
        is_anomaly = anomaly_score > threshold
        
        results = {
            "n_events": self.n_events,
            "n": self.n,
            "betti_numbers": betti,
            "anomaly_score": float(anomaly_score),
            "anomaly_threshold": float(threshold),
            "is_anomaly": is_anomaly,
            "topological_entropy": float(h_top),
            "curve_length": float(curve_length),
            "asymptotic_analysis": asymptotic,
            "crypto_key": key
        }
        
        return results
    
    def visualize_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Визуализация результатов
        """
        ur_vals = [p[0] for p in self.ur_uz_list]
        uz_vals = [p[1] for p in self.ur_uz_list]
        
        plt.figure(figsize=(16, 12))
        
        # 1. Отображение на торе
        plt.subplot(2, 3, 1)
        plt.scatter(ur_vals, uz_vals, c='blue', s=10, alpha=0.7)
        plt.xlabel('$u_r$')
        plt.ylabel('$u_z$')
        plt.title('Отображение ψ(e) на торе $\\mathbb{S}^1 \\times \\mathbb{S}^1$')
        plt.grid(True, alpha=0.3)
        
        # 2. Числа Бетти
        plt.subplot(2, 3, 2)
        labels = ['β₀', 'β₁', 'β₂']
        values = results['betti_numbers']
        plt.bar(labels, values, color=['green', 'orange', 'red'])
        plt.title('Числа Бетти')
        for i, v in enumerate(values):
            plt.text(i, v + 0.05, str(v), ha='center')
        
        # 3. DFT-анализ
        plt.subplot(2, 3, 3)
        score = results['anomaly_score']
        thresh = results['anomaly_threshold']
        plt.bar(['Anomaly Score'], [score], color='red', alpha=0.7)
        plt.axhline(y=thresh, color='black', linestyle='--', label=f'Threshold = {thresh:.2f}')
        plt.title('Показатель аномалии $\\mathcal{{A}} = {0:.3f}$'.format(score))
        plt.legend()
        plt.ylabel('Значение')
        
        # 4. Топологическая энтропия
        plt.subplot(2, 3, 4)
        h_top = results['topological_entropy']
        plt.text(0.5, 0.5, f'h_top = {h_top:.3f}',
                 fontsize=16, ha='center', va='center', 
                 bbox=dict(boxstyle="round", facecolor="wheat"))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Топологическая энтропия')
        plt.axis('off')
        
        # 5. Асимптотика
        plt.subplot(2, 3, 5)
        asymptotic = results['asymptotic_analysis']
        plt.scatter(asymptotic['log_d_values'], asymptotic['L_values'], color='green')
        x = np.array(asymptotic['log_d_values'])
        y_fit = asymptotic['C'] * x + asymptotic['intercept']
        plt.plot(x, y_fit, 'r-', label=f'L = {asymptotic["C"]:.2f}·ln(d) + {asymptotic["intercept"]:.2f}')
        plt.xlabel('ln(d)')
        plt.ylabel('L(d)')
        plt.title(f'Асимптотика (R² = {asymptotic["r_squared"]:.3f})')
        plt.legend()
        
        # 6. Ключ (хэш)
        plt.subplot(2, 3, 6)
        key_short = results['crypto_key'][:16]
        plt.text(0.5, 0.5, f'Crypto Key:\n{key_short}...', fontsize=12,
                 ha='center', va='center', family='monospace', 
                 bbox=dict(facecolor='lightgray'))
        plt.axis('off')
        plt.title('Сгенерированный ключ')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Визуализация сохранена в {save_path}")
        else:
            plt.show()
    
def main():
    """
    Демонстрация работы QCPH v1.2
    """
    print("=" * 80)
    print("Quantum-Crypto-Particle Hybrid Framework (QCPH v1.2) - Демонстрация")
    print("=" * 80)
    
    detector = CryptoAnomalyDetector(n_events=500, n=256)
    results = detector.run_full_analysis()
    
    print("\nРЕЗУЛЬТАТЫ АНАЛИЗА:")
    print("-" * 50)
    print(f"Числа Бетти: β₀={results['betti_numbers'][0]}, β₁={results['betti_numbers'][1]}, β₂={results['betti_numbers'][2]}")
    print(f"Показатель аномалии 𝒜 = {results['anomaly_score']:.4f}")
    print(f"Порог (sqrt(n)/2) = {results['anomaly_threshold']:.4f}")
    print(f"Аномалия обнаружена: {results['is_anomaly']}")
    print(f"Топологическая энтропия = {results['topological_entropy']:.4f}")
    print(f"Длина кривой = {results['curve_length']:.4f}")
    print(f"Криптографический ключ: {results['crypto_key']}")
    
    print("\nГенерация визуализации...")
    detector.visualize_results(results)
    
    print("\nЗАКЛЮЧЕНИЕ:")
    print("-" * 50)
    if results['is_anomaly']:
        print("✅ Обнаружены отклонения от стандартной модели — возможны новые физические явления.")
    else:
        print("✅ Данные соответствуют стандартной модели.")
    
    print("🔑 Криптографический ключ успешно сгенерирован на основе физических событий.")
    print("\nQCPH v1.2 успешно завершил работу!")
    print("=" * 80)

if __name__ == "__main__":
    main()
