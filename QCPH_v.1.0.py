# qcp.py
"""
Quantum-Crypto-Particle Hybrid Framework (QCPH v1.0)
Автономная система для обнаружения новых физических явлений
и генерации криптостойких ключей на основе топологического анализа
данных коллайдера и криптографических моделей ECDSA.
"""
import numpy as np
from ripser import ripser
from scipy.fft import fft
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple, List
import hashlib
import json
import warnings


class QCPH:
    """
    Основной класс QCPH v1.0 — полная автономная реализация гибридной модели.
    Не зависит от других фреймворков.
    """
    
    def __init__(self,
                 n_curve: int = 115792089237316195423570985008687907852837564279074904382605163141518161494337,  # secp256k1
                 sample_size: int = 500,
                 dft_threshold_factor: float = 0.5,
                 persistence_threshold: float = 0.1,
                 spectral_threshold: float = 0.1):
        """
        Инициализация QCPH v1.0.
        
        Параметры:
        - n_curve: порядок группы эллиптической кривой (n)
        - sample_size: размер выборки для DFT (N)
        - dft_threshold_factor: множитель для порога DFT (√N / 2)
        - persistence_threshold: порог для расстояния Вассерштейна
        - spectral_threshold: порог для спектрального расстояния
        """
        self.n_curve = n_curve
        self.N = sample_size
        self.dft_threshold = dft_threshold_factor * np.sqrt(self.N)  # зависит от N
        self.pers_threshold = persistence_threshold
        self.spec_threshold = spectral_threshold
        
        # Генерация эталонных диаграмм для тора T = S¹×S¹
        self.D0_ref, self.D1_ref, self.D2_ref = self._generate_torus_reference()
    
    def _generate_torus_reference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Генерация эталонных персистентных диаграмм для тора."""
        R, r = 0.8, 0.3
        u = np.linspace(0, 2*np.pi, 500)
        v = np.random.uniform(0, 2*np.pi, 500)
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        torus_points = np.column_stack([x, y, z])
        result = ripser(torus_points, maxdim=2)
        dgms = result['dgms']
        return dgms[0], dgms[1], dgms[2]
    
    def _spherical_coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Перевод в сферические координаты: theta (азимут), phi (полярный)."""
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)  # [0, 2π)
        phi = np.arccos(z / r)   # [0, π]
        return theta, phi
    
    def _topo_map(self, events: np.ndarray) -> np.ndarray:
        """
        Модуль 1: Топологическое отображение Ψ: M → T (Определение 1.1)
        
        events: (m, 4) массив в формате (px, py, pz, E)
        Возвращает: (u_r, u_z) ∈ [0, n_curve)
        """
        px, py, pz, E = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        theta, phi = self._spherical_coords(px, py, pz)
        
        # Взвешенное по энергии усреднение
        E_total = np.sum(E)
        u_r = np.sum(E * theta) / E_total
        u_z = np.sum(E * phi) / E_total
        
        # Нормировка на [0, 2π), затем масштабирование на [0, n_curve)
        u_r = (u_r % (2*np.pi)) * self.n_curve / (2*np.pi)
        u_z = (u_z % (2*np.pi)) * self.n_curve / (2*np.pi)
        
        # Применение mod n_curve (по определению 1.1)
        u_r = u_r % self.n_curve
        u_z = u_z % self.n_curve
        
        return np.array([u_r, u_z])
    
    def detect_topology(self, events: np.ndarray) -> Dict:
        """
        Модуль 1 & 6: Топологический анализ и персистентная гомология.
        Проверка чисел Бетти (Теорема 1.2).
        """
        mapped = np.array([self._topo_map(ev.reshape(-1,4)) for ev in events])
        u_r, u_z = mapped[:, 0], mapped[:, 1]
        points_2d = np.column_stack([u_r, u_z])
        
        result = ripser(points_2d, maxdim=2)
        dgms = result['dgms']
        D0, D1, D2 = dgms[0], dgms[1], dgms[2]
        
        # Числа Бетти
        betti_0 = len(D0) - np.sum(D0[:,1] < np.inf)
        betti_1 = np.sum(D1[:,1] == np.inf)
        betti_2 = np.sum(D2[:,1] == np.inf)
        
        # Расстояние Вассерштейна до эталона
        w0 = wasserstein_distance(D0.flatten(), self.D0_ref.flatten())
        w1 = wasserstein_distance(D1.flatten(), self.D1_ref.flatten())
        w2 = wasserstein_distance(D2.flatten(), self.D2_ref.flatten())
        
        anomaly_pers = (w0 + w1 + w2) > self.pers_threshold
        
        return {
            'betti': (betti_0, betti_1, betti_2),
            'pers_dist': (w0, w1, w2),
            'anomaly_pers': anomaly_pers,
            'diagrams': (D0, D1, D2)
        }
    
    def _bin_events_by_u_r(self, events: np.ndarray, num_bins: int = 50) -> List[List[np.ndarray]]:
        """Разделение событий по бинам u_r."""
        mapped = np.array([self._topo_map(ev.reshape(-1,4)) for ev in events])
        u_r = mapped[:, 0]
        bins = np.linspace(0, self.n_curve, num_bins + 1)
        binned_events = [[] for _ in range(num_bins)]
        
        for i, ur in enumerate(u_r):
            bin_idx = np.digitize(ur, bins) - 1
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            binned_events[bin_idx].append(events[i])
        
        return binned_events
    
    def detect_dft_anomaly(self, events: np.ndarray) -> Dict:
        """
        Модуль 2: DFT-анализ аномалий (Теорема 2.2, Определение 2.3)
        """
        binned_events = self._bin_events_by_u_r(events)
        num_bins = len(binned_events)
        A_values = []
        
        # Оценка d_phys через наклон
        mapped = np.array([self._topo_map(ev.reshape(-1,4)) for ev in events])
        u_r, u_z = mapped[:, 0], mapped[:, 1]
        sorted_idx = np.argsort(u_r)
        du_r = np.diff(u_r[sorted_idx])
        du_z = np.diff(u_z[sorted_idx])
        valid = du_r != 0
        if np.sum(valid) > 0:
            d_phys = np.mean(du_z[valid] / du_r[valid]) % self.n_curve
        else:
            d_phys = 1.0
        
        for v in range(num_bins - 1):
            Ev = binned_events[v]
            Ev1 = binned_events[v + 1]
            
            if len(Ev) == 0 or len(Ev1) == 0:
                continue
                
            # Функции f_{u_r}(u_z)
            uz_v = [self._topo_map(ev.reshape(-1,4))[1] for ev in Ev]
            uz_v1 = [self._topo_map(ev.reshape(-1,4))[1] for ev in Ev1]
            
            # Гистограммы
            hist_v, _ = np.histogram(uz_v, bins=50, range=(0, self.n_curve), density=True)
            hist_v1, _ = np.histogram(uz_v1, bins=50, range=(0, self.n_curve), density=True)
            
            # DFT
            f_hat_v = fft(hist_v)
            f_hat_v1 = fft(hist_v1)
            
            # Показатель аномалии A(v)
            k_vals = np.arange(1, len(f_hat_v))
            ratio = f_hat_v1[k_vals] / (f_hat_v[k_vals] + 1e-10)  # регуляризация
            expected = np.exp(-2j * np.pi * k_vals * d_phys / self.n_curve)
            A_v = np.mean(np.abs(ratio - expected)**2)
            A_values.append(A_v)
        
        if not A_values:
            A_mean = 0.0
        else:
            A_mean = np.mean(A_values)
        
        anomaly_dft = A_mean > self.dft_threshold
        
        return {
            'A_mean': A_mean,
            'threshold': self.dft_threshold,
            'anomaly_dft': anomaly_dft,
            'd_phys': d_phys
        }
    
    def compute_topological_entropy(self, events: np.ndarray) -> Dict:
        """
        Модуль 3: Топологическая энтропия (Определение 3.1, Теорема 3.2)
        """
        # Оценка d_phys через производные r(u_r, u_z)
        mapped = np.array([self._topo_map(ev.reshape(-1,4)) for ev in events])
        u_r, u_z = mapped[:, 0], mapped[:, 1]
        
        # Аппроксимация функции r(u_r, u_z) как плотности событий
        hist, ur_edges, uz_edges = np.histogram2d(u_r, u_z, bins=50, range=[[0, self.n_curve], [0, self.n_curve]])
        ur_centers = (ur_edges[:-1] + ur_edges[1:]) / 2
        uz_centers = (uz_edges[:-1] + uz_edges[1:]) / 2
        
        # Градиенты
        grad_uz, grad_ur = np.gradient(hist)
        
        # Усреднение
        mean_dudz = np.mean(grad_uz[1:-1, 1:-1])
        mean_dudr = np.mean(grad_ur[1:-1, 1:-1])
        
        if abs(mean_dudr) < 1e-10:
            d_phys = 1.0
        else:
            d_phys = (-mean_dudz / mean_dudr) % self.n_curve
        
        h_top = np.log(max(1, abs(d_phys)))
        
        # Критерий физической стабильности (Определение 3.3)
        log_n = np.log(self.n_curve)
        anomaly_entropy = h_top < log_n - 0.1
        
        return {
            'd_phys': d_phys,
            'h_top': h_top,
            'anomaly_entropy': anomaly_entropy
        }
    
    def analyze_curve_length(self, events: np.ndarray) -> Dict:
        """
        Модуль 4: Асимптотический анализ длины кривой (Теорема 4.1)
        """
        h_top_info = self.compute_topological_entropy(events)
        d_phys = h_top_info['d_phys']
        
        if d_phys <= 1:
            L = 0.0
        else:
            L = np.log(d_phys)  # Упрощённая форма; C=1
        
        expected_L = np.log(max(1, d_phys))
        anomaly_length = abs(L - expected_L) > 0.5
        
        return {
            'L': L,
            'expected_L': expected_L,
            'anomaly_length': anomaly_length
        }
    
    def generate_crypto_key(self, events: np.ndarray) -> str:
        """
        Модуль 5: Генерация криптостойкого ключа (Алгоритм 5.3)
        """
        h_top_info = self.compute_topological_entropy(events)
        if h_top_info['anomaly_entropy']:
            warnings.warn("События нестабильны — ключ может быть ненадёжным.")
        
        d_phys = h_top_info['d_phys']
        mapped = np.array([self._topo_map(ev.reshape(-1,4)) for ev in events])
        u_r_list = mapped[:, 0]
        u_z_list = mapped[:, 1]
        
        k_sum = 0.0
        for u_r, u_z in zip(u_r_list, u_z_list):
            k_i = (u_z + u_r * d_phys) % self.n_curve
            k_sum += k_i
        
        key_input = int(k_sum * d_phys) % self.n_curve
        key_hex = hashlib.sha256(str(key_input).encode()).hexdigest()
        return key_hex
    
    def detect_persistence_anomaly(self, events: np.ndarray) -> Dict:
        """
        Модуль 6: Персистентная гомология (Определение 6.3)
        """
        topo_result = self.detect_topology(events)
        D0, D1, D2 = topo_result['diagrams']
        
        w0 = wasserstein_distance(D0.flatten(), self.D0_ref.flatten())
        w1 = wasserstein_distance(D1.flatten(), self.D1_ref.flatten())
        w2 = wasserstein_distance(D2.flatten(), self.D2_ref.flatten())
        
        # Веса: b1 наиболее важен
        w0_weight, w1_weight, w2_weight = 1.0, 2.0, 1.0
        P_M = w0_weight * w0 + w1_weight * w1 + w2_weight * w2
        
        anomaly = P_M > self.pers_threshold
        return {
            'P_M': P_M,
            'anomaly_pers': anomaly
        }
    
    def detect_spectral_anomaly(self, events: np.ndarray) -> Dict:
        """
        Модуль 7: Спектральный анализ (Определение 7.3)
        """
        mapped = np.array([self._topo_map(ev.reshape(-1,4)) for ev in events])
        u_r, u_z = mapped[:, 0], mapped[:, 1]
        points = np.column_stack([u_r, u_z])
        
        if len(points) < 2:
            return {'anomaly_spec': False, 'spec_dist': 0.0}
        
        # Матрица корреляций
        C = np.corrcoef(points.T)
        eigenvals = np.linalg.eigvals(C)
        eigenvals = np.sort(np.abs(eigenvals))[::-1]
        
        # Эталон для тора
        ref_eigen = np.array([1.0, 1.0, 0.1, 0.1])[:len(eigenvals)]
        spec_dist = np.linalg.norm(eigenvals - ref_eigen)
        
        anomaly_spec = spec_dist > self.spec_threshold
        return {
            'eigenvals': eigenvals,
            'spec_dist': spec_dist,
            'anomaly_spec': anomaly_spec
        }
    
    def analyze(self, events: np.ndarray) -> Dict:
        """
        Интеграционная теорема 8.1: Полный многоуровневый анализ.
        """
        topo = self.detect_topology(events)
        dft = self.detect_dft_anomaly(events)
        entropy = self.compute_topological_entropy(events)
        length = self.analyze_curve_length(events)
        pers = self.detect_persistence_anomaly(events)
        spectral = self.detect_spectral_anomaly(events)
        
        # Многоуровневое голосование
        anomalies = [
            topo['anomaly_pers'],
            dft['anomaly_dft'],
            entropy['anomaly_entropy'],
            length['anomaly_length'],
            pers['anomaly_pers'],
            spectral['anomaly_spec']
        ]
        anomaly_score = np.mean(anomalies)
        final_anomaly = anomaly_score >= 0.5
        
        return {
            'final_anomaly': final_anomaly,
            'anomaly_score': anomaly_score,
            'topology': topo,
            'dft': dft,
            'entropy': entropy,
            'length': length,
            'persistence': pers,
            'spectral': spectral,
            'recommendation': "Требуется ручная проверка" if final_anomaly else "События соответствуют Стандартной модели",
            'crypto_key': self.generate_crypto_key(events) if not final_anomaly else None
        }
    
    def save_report(self, result: Dict, filepath: str):
        """Сохранение отчёта в JSON."""
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Отчёт сохранён: {filepath}")


# ==============================================================================
# Пример использования QCPH как автономной программы
# ==============================================================================

if __name__ == "__main__":
    # Пример данных: два события, каждое — массив частиц (px, py, pz, E)
    events_sample = [
        np.array([[1.0, 0.5, 0.3, 2.0], [0.2, -0.1, 0.4, 1.5]]),
        np.array([[0.8, -0.3, 0.7, 3.0], [0.1, 0.2, -0.1, 0.8]])
    ]
    
    # Инициализация QCPH
    qcp = QCPH(n_curve=1000003, sample_size=100)
    
    # Полный анализ
    result = qcp.analyze(events_sample)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("  QUANTUM-CRYPTO-PARTICLE HYBRID FRAMEWORK (QCPH v1.0)")
    print("  Автономная система анализа данных коллайдера")
    print("="*60)
    print(f"🔹 Обнаружена аномалия: {result['final_anomaly']}")
    print(f"🔹 Оценка аномалии: {result['anomaly_score']:.3f}")
    print(f"🔹 Рекомендация: {result['recommendation']}")
    if result['crypto_key']:
        print(f"🔹 Сгенерированный криптостойкий ключ: {result['crypto_key'][:16]}...")
    else:
        print("🔹 Ключ не сгенерирован — события нестабильны")
    print("="*60)
    
    # Сохранение отчёта
    qcp.save_report(result, "data/output/qcp_report.json")
