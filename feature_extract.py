# 提取拟合曲线的所有特征
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction.feature_calculators import sample_entropy


def extract_features_from_fitted_curve(x_smooth, y_smooth):
    # 几何特征
    length = compute_length(x_smooth, y_smooth)
    mean_slope, max_slope, min_slope = compute_slopes(x_smooth, y_smooth)
    mean_curvature, max_curvature, min_curvature = compute_curvature(x_smooth, y_smooth)
    inflection_points = compute_inflection_points(x_smooth, y_smooth)
    # 统计特征
    max_val = np.max(y_smooth)
    min_val = np.min(y_smooth)
    mean_val = np.mean(y_smooth)
    std_val = np.std(y_smooth)
    range_val = max_val - min_val
    mid_val = np.median(y_smooth)
    total_variation = np.sum(np.abs(np.diff(y_smooth)))
    peaks, _ = find_peaks(y_smooth)  # 峰值
    valleys, _ = find_peaks(-y_smooth)  # 谷值
    num_peaks = len(peaks)
    num_valleys = len(valleys)
    # 频域特征
    fft_features = extract_fft_features(y_smooth, num_features=10)

    # 综合特征
    features = [
        length, mean_slope, max_slope, min_slope, mid_val, total_variation, num_peaks, num_valleys,
        mean_curvature, max_curvature, min_curvature, inflection_points,
        max_val, min_val, mean_val, std_val, range_val,
        compute_hurst_exponent(y_smooth),
        sample_entropy(y_smooth),
        feature_calculators.autocorrelation(y_smooth, lag=1)
    ]
    features.extend(fft_features)

    return np.array(features)


# 计算拟合曲线的特征
def compute_length(x, y):
    diff = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.sum(diff)


def compute_slopes(x, y):
    slopes = np.diff(y) / np.diff(x)
    return np.mean(slopes), np.max(slopes), np.min(slopes)


def compute_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2 + 1e-6)
    return np.mean(curvature), np.max(curvature), np.min(curvature)


def compute_inflection_points(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = dx * ddy - dy * ddx
    inflection_points = np.where(np.diff(np.sign(curvature)))[0]
    return len(inflection_points)


def extract_fft_features(y, num_features=12):
    fft_vals = fft(y)
    fft_magnitudes = np.abs(fft_vals)
    return fft_magnitudes[:num_features]

def extract_all_curves(x_resampled, fitted_curves):
    all_features = []
    for curve_idx in range(len(fitted_curves)):
        # 生成平滑曲线数据
        x_smooth = np.linspace(x_resampled[:, curve_idx].min(), x_resampled[:, curve_idx].max(), 500)
        y_smooth = fitted_curves[curve_idx](x_smooth)
        # 提取特征
        features = extract_features_from_fitted_curve(x_smooth, y_smooth)
        all_features.append(features)

    # 转为 NumPy 数组
    all_features = np.array(all_features)
    return all_features

import numpy as np


def compute_hurst_exponent(y, max_lag=20):
    """
    计算 Hurst 指数
    :param y: 输入序列
    :param max_lag: 最大滞后阶数
    :return: Hurst 指数
    """
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(y[lag:], y[:-lag])) for lag in lags]
    hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return hurst
