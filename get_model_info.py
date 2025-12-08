from ultralytics import YOLO
import torch

model = YOLO('data/LLVIP_IF-yolo11x-e300-16-pretrained.pt')

# 检查模型配置
if hasattr(model.model, 'yaml'):
    yaml = model.model.yaml
    print('输入通道数:', yaml.get('ch', '未知'))
    print('类别数:', yaml.get('nc', '未知'))
    print(f"  YAML来源文件: {yaml.get('yaml_file', '未记录')}")
    print('Backbone前3层:')
    for i, layer in enumerate(yaml.get('backbone', [])[:3]):
        print(f'  {i}: {layer}')
else:
    print('无法获取YAML配置')

# 检查第一层卷积
first_conv = None
for m in model.model.modules():
    if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
        first_conv = m.conv
        break

if first_conv:
    print(f'\n第一层卷积权重形状: {first_conv.weight.shape}')
    print(f'输入通道数: {first_conv.weight.shape[1]}')