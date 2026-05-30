"""
VMD Processor Module
====================
Implementasi Variational Mode Decomposition (VMD) untuk dekomposisi
sinyal GNSS menjadi Intrinsic Mode Functions (IMF).

Referensi: Dragomiretskiy & Zosso (2013)
  "Variational Mode Decomposition", IEEE Trans. Signal Processing

Dua pilihan backend:
  1. vmdpy  (pip install vmdpy) — implementasi referensi
  2. Implementasi lokal (fallback) — mengikuti formulasi asli di domain Fourier
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VMDConfig:
    """Konfigurasi untuk VMD."""
    n_modes:        int   = 5        # Jumlah mode IMF (K)
    alpha:          float = 2000.0   # Balancing parameter (bandwidth)
    tau:            float = 0.0      # Time-step dual ascent (0 = noise-tolerant)
    DC:             int   = 0        # Apakah IMF-1 = DC component
    init:           int   = 1        # Inisialisasi frekuensi (1=uniform, 0=zero)
    tol:            float = 1e-7     # Konvergensi tolerance
    max_iter:       int   = 500      # Iterasi maksimum


def _vmd_core(signal: np.ndarray, alpha: float, tau: float,
               K: int, DC: int, init: int, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementasi VMD dalam domain Fourier.
    Mengikuti Algorithm 1 dari Dragomiretskiy & Zosso (2013).

    Args:
        signal : 1D input signal (panjang T)
        alpha  : bandwidth constraint
        tau    : noise-tolerance (0 = Lagrangian, >0 = augmented)
        K      : jumlah mode
        DC     : apakah mode pertama = DC
        init   : 1=uniform freq init, 0=random
        tol    : konvergensi

    Returns:
        u      : (K, T) IMF dalam domain waktu
        omega  : (K,)  center frequencies (normalized 0..0.5)
    """
    T = len(signal)

    # Mirror signal untuk mengurangi end-effects
    T2 = T // 2
    f_mirror = np.concatenate([signal[T2:0:-1], signal, signal[-1:T2:-1]])
    T_ext = len(f_mirror)

    # Frekuensi domain (positif saja, normalized 0..0.5)
    freqs = np.arange(T_ext) / T_ext - 0.5
    freqs = np.fft.fftshift(freqs)
    # Hanya setengah positif
    freqs_half = freqs[T_ext // 2:]

    # FFT dari sinyal
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T_ext // 2] = 0  # Hanya frekuensi positif

    # Inisialisasi mode Fourier dan frekuensi pusat
    u_hat = np.zeros((K, T_ext), dtype=complex)
    omega_k = np.zeros(K)

    if init == 1:
        # Inisialisasi uniform
        for i in range(K):
            omega_k[i] = (0.5 / K) * i
    else:
        # Inisialisasi random (sorted)
        omega_k = np.sort(np.random.rand(K) * 0.5)

    if DC:
        omega_k[0] = 0.0

    # Dual variable (Lagrangian)
    lambda_hat = np.zeros(T_ext, dtype=complex)

    # Iterasi utama
    uDiff = tol + 1
    n_iter = 0
    u_hat_old = u_hat.copy()

    while uDiff > tol and n_iter < 500:
        u_hat_old = u_hat.copy()

        # Update setiap mode k
        for k in range(K):
            # Hitung akumulasi mode lain
            sum_others = np.zeros(T_ext, dtype=complex)
            for j in range(K):
                if j != k:
                    sum_others += u_hat[j]

            # Update u_hat[k] dalam domain Fourier
            num = f_hat_plus - sum_others - lambda_hat / 2.0
            den = 1.0 + 2.0 * alpha * (freqs - omega_k[k]) ** 2
            u_hat[k] = num / den

            # Update omega_k[k] (center frequency)
            if DC and k == 0:
                omega_k[0] = 0.0
            else:
                abs2 = np.abs(u_hat[k][T_ext // 2:]) ** 2
                freqs_p = freqs[T_ext // 2:]
                denom = np.sum(abs2)
                if denom > 1e-12:
                    omega_k[k] = np.dot(freqs_p, abs2) / denom

        # Update dual variable
        lambda_hat += tau * (np.sum(u_hat, axis=0) - f_hat_plus)

        # Konvergensi check
        uDiff = np.sum(np.abs(u_hat - u_hat_old) ** 2) / (T_ext + 1e-12)
        n_iter += 1

    logger.debug(f"VMD converged at iteration {n_iter} (uDiff={uDiff:.2e})")

    # Rekonstruksi domain waktu dari seluruh spektrum (conjugate symmetry)
    u = np.zeros((K, T_ext))
    for k in range(K):
        u_hat_full = np.zeros(T_ext, dtype=complex)
        u_hat_full[T_ext // 2:] = u_hat[k][T_ext // 2:]
        u_hat_full[1:T_ext // 2] = np.conj(u_hat[k][T_ext // 2 + 1:][::-1])
        u[k] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat_full)))

    # Potong kembali ke panjang asli (hapus mirror)
    u = u[:, T2:T2 + T]

    return u, omega_k


class VMDProcessor:
    """Dekomposisi sinyal menggunakan VMD."""

    def __init__(self, config: Optional[VMDConfig] = None):
        self.config = config or VMDConfig()
        # Coba import vmdpy untuk implementasi yang lebih robust
        try:
            from vmdpy import VMD as vmdpy_VMD
            self._use_vmdpy = True
            self._vmdpy_VMD = vmdpy_VMD
            logger.info(f"VMDProcessor: menggunakan vmdpy backend")
        except ImportError:
            self._use_vmdpy = False
            logger.info(f"VMDProcessor: menggunakan implementasi lokal "
                        f"(install vmdpy untuk performa lebih baik)")
        logger.info(f"  K={self.config.n_modes}, alpha={self.config.alpha}, "
                    f"tol={self.config.tol}")

    def vmd(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dekomposisi VMD.

        Args:
            signal: 1D time series (T,)

        Returns:
            imfs  : (K, T) IMF
            omega : (K,) center frequencies (normalized)
        """
        # Tangani NaN
        nan_mask = np.isnan(signal)
        if nan_mask.any():
            signal = signal.copy()
            valid = ~nan_mask
            # Interpolasi linear sederhana untuk NaN sebelum VMD
            xi = np.where(valid)[0]
            xp = np.arange(len(signal))
            signal = np.interp(xp, xi, signal[xi])

        # Normalisasi sinyal (VMD sensitif terhadap skala)
        sig_mean = np.mean(signal)
        sig_std = np.std(signal)
        if sig_std < 1e-12:
            sig_std = 1.0
        signal_norm = (signal - sig_mean) / sig_std

        if self._use_vmdpy:
            try:
                imfs, _, omega = self._vmdpy_VMD(
                    signal_norm,
                    alpha=self.config.alpha,
                    tau=self.config.tau,
                    K=self.config.n_modes,
                    DC=self.config.DC,
                    init=self.config.init,
                    tol=self.config.tol
                )
                # Denormalisasi
                imfs = imfs * sig_std
                # Tambahkan mean ke mode pertama (bias)
                imfs[0] += sig_mean
                return imfs, omega
            except Exception as e:
                logger.warning(f"vmdpy gagal: {e}, fallback ke implementasi lokal")

        # Implementasi lokal
        imfs, omega = _vmd_core(
            signal_norm,
            alpha=self.config.alpha,
            tau=self.config.tau,
            K=self.config.n_modes,
            DC=self.config.DC,
            init=self.config.init,
            tol=self.config.tol
        )
        # Denormalisasi
        imfs = imfs * sig_std
        imfs[0] += sig_mean

        return imfs, omega

    def decompose_signal(self, signal: np.ndarray) -> Dict:
        """
        Dekomposisi + hitung residual.

        Returns dict dengan keys:
            'imfs'          : (K, T) IMF
            'omega'         : (K,) center frequencies
            'reconstruction': (T,) rekonstruksi dari semua IMF
            'residual'      : (T,) sisa (signal - reconstruction)
        """
        imfs, omega = self.vmd(signal)
        reconstruction = np.sum(imfs, axis=0)
        residual = signal - reconstruction

        recon_err = np.sqrt(np.mean(residual ** 2))
        logger.debug(f"VMD reconstruction RMSE: {recon_err:.6e}")

        return {
            'imfs': imfs,
            'omega': omega,
            'reconstruction': reconstruction,
            'residual': residual
        }

    @staticmethod
    def reconstruct_from_imfs(imfs: np.ndarray,
                               indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Rekonstruksi sinyal dari IMF (semua atau subset).

        Args:
            imfs   : (K, T)
            indices: list indeks IMF yang dipakai (None = semua)
        """
        if indices is not None:
            return np.sum(imfs[indices], axis=0)
        return np.sum(imfs, axis=0)

    @staticmethod
    def select_by_energy(imfs: np.ndarray,
                          threshold: float = 0.99) -> Tuple[np.ndarray, List[int]]:
        """
        Pilih IMF yang secara kolektif menyumbang ≥ threshold energi total.

        Returns:
            selected_imfs : (K', T)
            selected_idx  : list indeks
        """
        energy = np.sum(imfs ** 2, axis=1)
        total = energy.sum()
        if total < 1e-12:
            return imfs, list(range(len(imfs)))

        order = np.argsort(-energy)
        cum = 0.0
        selected = []
        for idx in order:
            cum += energy[idx] / total
            selected.append(int(idx))
            if cum >= threshold:
                break

        selected.sort()
        logger.info(f"Dipilih {len(selected)}/{len(imfs)} IMF "
                    f"(energy coverage {cum*100:.1f}%)")
        return imfs[selected], selected


class PipelineVMD:
    """Wrapper VMD untuk pipeline multi-stasiun."""

    def __init__(self, config: Optional[VMDConfig] = None):
        self.config = config or VMDConfig()
        self.processor = VMDProcessor(self.config)
        self.decompositions: Dict[str, Dict[str, Dict]] = {}

    def decompose_all_stations(self,
                                data_matrix: np.ndarray,
                                station_names: List[str]) -> Dict[str, Dict[str, Dict]]:
        """
        Dekomposisi VMD untuk semua stasiun × 3 axis.

        Args:
            data_matrix  : (n_epochs, n_stations, 3)
            station_names: list nama stasiun

        Returns:
            {station_name: {'E': decomp_dict, 'N': ..., 'U': ...}}
        """
        n_epochs, n_stations, n_axes = data_matrix.shape
        axes = ['E', 'N', 'U']

        logger.info(f"Memulai VMD decomposition: {n_stations} stasiun × 3 axis")

        for s_idx, name in enumerate(station_names):
            logger.info(f"[{s_idx+1}/{n_stations}] {name}")
            station_decomps = {}

            for a_idx, axis in enumerate(axes):
                signal = data_matrix[:, s_idx, a_idx]
                try:
                    decomp = self.processor.decompose_signal(signal)
                    station_decomps[axis] = decomp
                    rmse = np.sqrt(np.mean(decomp['residual'] ** 2))
                    logger.info(f"  {axis}: {self.config.n_modes} IMF, "
                                f"reconstruction RMSE={rmse:.6e}")
                except Exception as e:
                    logger.warning(f"  {axis}: VMD gagal — {e}")
                    station_decomps[axis] = {
                        'imfs': np.zeros((self.config.n_modes, n_epochs)),
                        'omega': np.zeros(self.config.n_modes),
                        'reconstruction': signal,
                        'residual': np.zeros(n_epochs)
                    }

            self.decompositions[name] = station_decomps

        return self.decompositions

    def get_imf_feature_matrix(self) -> np.ndarray:
        """
        Susun semua IMF menjadi feature matrix untuk input CNN-LSTM.

        Returns:
            (n_epochs, n_stations × n_axes × K)
        """
        feature_lists = []
        for station_decomps in self.decompositions.values():
            for decomp in station_decomps.values():
                # imfs: (K, T) → transposed ke (T, K)
                feature_lists.append(decomp['imfs'].T)

        # Stack: (T, n_stations*n_axes*K)
        feature_matrix = np.concatenate(feature_lists, axis=1)
        logger.info(f"Feature matrix dari VMD: {feature_matrix.shape}")
        return feature_matrix


def main_vmd():
    """Test VMD dengan sinyal sintetis."""
    np.random.seed(42)
    t = np.linspace(0, 2, 2000)
    signal = (np.sin(2 * np.pi * 3 * t) +
              0.5 * np.sin(2 * np.pi * 10 * t) +
              0.2 * np.sin(2 * np.pi * 25 * t) +
              0.05 * np.random.randn(len(t)))

    config = VMDConfig(n_modes=3, alpha=2000, tol=1e-7)
    processor = VMDProcessor(config)
    decomp = processor.decompose_signal(signal)

    print(f"Signal shape : {signal.shape}")
    print(f"IMFs shape   : {decomp['imfs'].shape}")
    print(f"Center freqs : {decomp['omega']}")
    rmse = np.sqrt(np.mean(decomp['residual'] ** 2))
    print(f"Reconstruction RMSE: {rmse:.6e}")


if __name__ == "__main__":
    main_vmd()
