import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def check_model_info():
    try:
        # 加载模型
        model_path = "data/LLVIP-yolo11m-e300-16-pretrained.pt"
        print(f"正在加载模型: {model_path}")
        
        model = YOLO(model_path)
        
        # 打印模型的基本信息
        print("\n模型基本信息:")
        print(f"模型架构: {model.__class__.__name__}")
        
        # 打印模型的配置信息
        print("\n模型配置信息:")
        if hasattr(model, 'model'):
            print(f"模型层数: {len(list(model.model.modules()))}")
        
        # 尝试访问模型的配置
        if hasattr(model, 'cfg'):
            print(f"模型配置: {model.cfg}")
        
        # 打印模型的详细信息
        print("\n模型详细信息:")
        model.info()
        
        # 查看模型的默认图像尺寸
        print(f"\n模型默认图像尺寸 (imgsz):")
        # 检查模型是否存储了训练时的图像尺寸
        if hasattr(model, 'task'):
            print(f"任务类型: {model.task}")
        
        # 尝试从模型的配置中查找图像尺寸
        if hasattr(model, 'overrides'):
            print(f"覆盖参数: {model.overrides}")
            if 'imgsz' in model.overrides:
                print(f"覆盖的图像尺寸: {model.overrides['imgsz']}")
        
        # 检查模型本身是否存储了图像尺寸信息
        if hasattr(model, 'model') and hasattr(model.model, 'imgsz'):
            print(f"模型内部图像尺寸: {model.model.imgsz}")
            
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        print("请确保模型文件存在于指定路径")
    except Exception as e:
        print(f"加载模型时发生错误: {str(e)}")

if __name__ == "__main__":
    check_model_info()