from pathlib import Path

def clean_unlabeled_images(data_dir):
    """
    删除没有对应标注文件的图片

    Args:
        data_dir: 数据目录路径，包含images和labels两个子目录
    """
    data_path = Path(data_dir)
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels'

    if not images_dir.exists():
        print(f"错误: images目录不存在: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"错误: labels目录不存在: {labels_dir}")
        return

    # 获取所有标注文件的文件名（不含扩展名）
    label_files = set()
    for label_file in labels_dir.rglob('*.txt'):
        label_files.add(label_file.stem)

    print(f"找到 {len(label_files)} 个标注文件")

    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

    # 遍历所有图片文件
    deleted_count = 0
    kept_count = 0

    for image_file in images_dir.rglob('*'):
        if image_file.is_file() and image_file.suffix in image_extensions:
            # 检查是否有对应的标注文件
            if image_file.stem not in label_files:
                print(f"删除无标注图片: {image_file.relative_to(data_path)}")
                image_file.unlink()
                deleted_count += 1
            else:
                kept_count += 1

    print(f"\n清理完成:")
    print(f"  保留图片: {kept_count}")
    print(f"  删除图片: {deleted_count}")

if __name__ == '__main__':
    # 指定数据目录路径
    data_directory = 'data/dataset/lake'

    print(f"开始清理数据集: {data_directory}")
    print("=" * 50)

    # 确认操作
    response = input(f"确认要删除 {data_directory} 中没有标注的图片吗? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        clean_unlabeled_images(data_directory)
    else:
        print("操作已取消")
