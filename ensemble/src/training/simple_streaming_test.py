"""
简单的流式推理测试
"""
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # 回到项目根目录
sys.path.append(str(project_root))

try:
    from ensemble.src.training.streaming_multi_rocket import StreamingMultiRocketClassifier, StreamingInferenceEngine
    print("成功导入流式模型模块")
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试直接导入tsai模块...")
    try:
        from tsai.models.HydraMultiRocketPlus import HydraMultiRocketBackbonePlus
        print("tsai模块可用")
    except ImportError:
        print("tsai模块不可用，请安装: pip install tsai")
        exit(1)

def simple_test():
    """简单测试"""
    print("开始简单测试...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 创建模型
        print("创建模型...")
        model = StreamingMultiRocketClassifier(
            c_in=28,
            c_out=4,
            max_seq_len=50,
            min_seq_len=10,
            device=device,
            confidence_threshold=0.8
        )
        print("模型创建成功！")
        print(f"支持的序列长度: {model.supported_lengths}")
        
        # 创建推理引擎
        print("创建推理引擎...")
        engine = StreamingInferenceEngine(model)
        print("推理引擎创建成功！")
        
        # 测试流式推理
        print("测试流式推理...")
        for t in range(20):
            features = np.random.randn(28)
            result = engine.add_timestep(features)
            
            if t % 5 == 0:
                print(f"时间步 {t+1}: 预测={result['prediction']}, 置信度={result['confidence']:.3f}")
            
            if result['stopped_early']:
                print(f"早期停止在时间步 {t+1}")
                break
        
        final_result = engine.get_final_prediction()
        print(f"最终结果: {final_result}")
        
        print("测试成功完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()