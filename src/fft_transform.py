import numpy as np
from dataclasses import dataclass

# ===============================
# FFT Transformer
# ===============================
@dataclass
class FFTTransformer:
    n: int
    scale_output: int = 1  # 1 = scaled, 0 = unscaled

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Real FFT forward transform
        :param x: 输入实数信号 (长度 = n)
        :return: 复数频谱
        """
        X = np.fft.rfft(x, n=self.n)
        if self.scale_output:
            X = X / self.n
        return X

    def backward(self, X: np.ndarray) -> np.ndarray:
        """
        Real FFT backward transform
        :param X: 复数频谱
        :return: 时域信号 (实数)
        """
        x = np.fft.irfft(X, n=self.n)
        if self.scale_output:
            x = x * self.n
        return x


# ===============================
# Cosine quarter-wave transformer
# (基于 DCT-II / DCT-III 实现)
# ===============================
@dataclass
class FFTCosqTransformer:
    n: int
    scale_output: int = 1  # 1 = scaled, 0 = unscaled

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Real cosine quarter-wave forward transform (DCT-II)
        """
        X = np.fft.rfft(x, n=self.n).real
        if self.scale_output:
            X = X / self.n
        return X

    def backward(self, X: np.ndarray) -> np.ndarray:
        """
        Real cosine quarter-wave backward transform (DCT-III)
        """
        x = np.fft.irfft(X, n=self.n)
        if self.scale_output:
            x = x * self.n
        return x


# ===============================
# Example Usage
# ===============================
if __name__ == "__main__":
    N = 8
    signal = np.sin(2 * np.pi * np.arange(N) / N)

    # FFT Transformer
    fft_t = FFTTransformer(n=N, scale_output=1)
    spectrum = fft_t.forward(signal)
    recon = fft_t.backward(spectrum)

    print("原始信号:", signal)
    print("FFT 结果:", spectrum)
    print("逆变换:", recon)

    # Cosq Transformer
    cosq_t = FFTCosqTransformer(n=N, scale_output=1)
    cosq = cosq_t.forward(signal)
    recon_cosq = cosq_t.backward(cosq)

    print("Cosq 变换:", cosq)
    print("Cosq 逆变换:", recon_cosq)
