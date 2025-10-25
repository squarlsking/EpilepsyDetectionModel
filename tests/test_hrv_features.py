"""
HRV特征提取模块测试
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.hrv_features import compute_hrv_features, compute_time_domain, compute_freq_domain


def test_time_domain():
    """测试时域特征提取"""
    print("测试时域特征...")
    
    # 模拟RR间期数据 (正常心率约60-100 bpm)
    rr_ms = np.array([800, 820, 810, 830, 825, 815, 805, 810, 820, 815])
    
    features = compute_time_domain(rr_ms)
    
    assert "meanNN" in features
    assert "sdnn" in features
    assert "rmssd" in features
    assert "pnn50" in features
    assert "sd1" in features
    assert "sd2" in features
    
    print(f"  ✓ meanNN: {features['meanNN']:.2f} ms")
    print(f"  ✓ sdnn: {features['sdnn']:.2f} ms")
    print(f"  ✓ rmssd: {features['rmssd']:.2f} ms")
    print(f"  ✓ pnn50: {features['pnn50']:.4f}")
    

def test_freq_domain():
    """测试频域特征提取"""
    print("\n测试频域特征...")
    
    # 生成足够长的RR数据
    rr_ms = np.random.normal(800, 50, 100)  # 100个RR间期
    
    features = compute_freq_domain(rr_ms)
    
    assert "lf" in features
    assert "hf" in features
    assert "lf_hf" in features
    
    print(f"  ✓ LF power: {features['lf']:.4f}")
    print(f"  ✓ HF power: {features['hf']:.4f}")
    print(f"  ✓ LF/HF ratio: {features['lf_hf']:.4f}")


def test_compute_hrv_features():
    """测试完整HRV特征提取"""
    print("\n测试完整HRV特征...")
    
    rr_ms = np.random.normal(800, 50, 100)
    features = compute_hrv_features(rr_ms)
    
    # 检查所有特征都存在
    expected_keys = ["meanNN", "sdnn", "rmssd", "pnn50", "sd1", "sd2", 
                     "sd2_sd1", "lf", "hf", "lf_hf", "lfn", "hfn"]
    
    for key in expected_keys:
        assert key in features, f"Missing feature: {key}"
    
    print(f"  ✓ 提取了 {len(features)} 个特征")
    print(f"  ✓ 特征列表: {list(features.keys())}")


def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")
    
    # 测试空数组
    features = compute_hrv_features(np.array([]))
    assert features == {}
    print("  ✓ 空数组处理正确")
    
    # 测试单个值
    features = compute_hrv_features(np.array([800]))
    assert features == {}
    print("  ✓ 单值处理正确")
    
    # 测试少量数据
    features = compute_hrv_features(np.array([800, 810]))
    assert "meanNN" in features
    print("  ✓ 少量数据处理正确")


if __name__ == "__main__":
    print("=" * 50)
    print("HRV特征提取测试套件")
    print("=" * 50)
    
    try:
        test_time_domain()
        test_freq_domain()
        test_compute_hrv_features()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
