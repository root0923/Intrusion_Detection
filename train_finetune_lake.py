import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练模型进行微调
    model = YOLO('ultralytics/cfg/models/11/yolo11m.yaml')

    # 微调训练配置
    model.train(
        data='ultralytics/cfg/datasets/lakeV.yaml',  # 数据集配置
        channels=3,           # 使用3通道RGB
        use_simotm='RGB',
        # 图像设置
        imgsz=800,
        cache=True,  # 如果内存够大可以设为True加速训练

        # 训练轮次（微调不需要太多轮）
        epochs=300,  # 微调建议30-100轮，根据数据量调整

        # 批次大小（根据显存调整）
        batch=32,  # 如果显存不够，可以改为8或4

        # 数据增强
        close_mosaic=10,  # 最后10轮关闭mosaic增强

        # 硬件设置
        workers=4,
        device='0',  # 单GPU，多GPU用'0,1'

        # 优化器设置（微调关键参数）
        optimizer='AdamW',  # AdamW对微调效果较好
        lr0=0.0001,  # 初始学习率：微调用0.0001-0.001
        lrf=0.01,  # 最终学习率系数：lr_final = lr0 * lrf
        momentum=0.937,  # SGD动量
        weight_decay=0.0005,  # 权重衰减，防止过拟合

        # 学习率调度
        warmup_epochs=3,  # 预热轮数
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 其他设置
        save=True,  # 保存检查点
        save_period=10,  # 每10轮保存一次

        # 输出目录
        project='runs/finetune_V_3classes',
        name='lake-yolo11m-finetune_V_3classes',
        exist_ok=False,  # 如果目录存在是否覆盖

        # 其他可选参数
        # pretrained=True,  # 是否使用预训练权重（加载.pt时自动为True）
        # resume=False,  # 是否从上次中断处继续训练
        # amp=True,  # 混合精度训练，加速训练
        # fraction=1.0,  # 使用数据集的比例（1.0=全部）
    )
