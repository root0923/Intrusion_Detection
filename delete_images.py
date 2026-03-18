import os
import glob

def delete_images_without_labels(images_dir='images', labels_dir='labels'):
    """
    删除images目录中没有对应标签文件的图片。

    参数:
        images_dir: 图片文件目录路径
        labels_dir: 标签文件目录路径
    """
    # 支持的图片扩展名（可按需增减）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

    # 获取labels目录下所有标签文件（通常是.txt文件）
    # 注意：这里获取标签文件的基本名（不含扩展名），用于匹配
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    label_basenames = {os.path.splitext(os.path.basename(f))[0] for f in label_files}

    print(f"标签目录中有 {len(label_basenames)} 个标签文件。")
    print("开始检查图片文件...")

    deleted_count = 0
    kept_count = 0

    # 遍历所有支持的图片格式
    for ext in image_extensions:
        image_files = glob.glob(os.path.join(images_dir, f"*{ext}"))
        for image_file in image_files:
            # 获取图片文件名（不含扩展名）
            image_basename = os.path.splitext(os.path.basename(image_file))[0]

            if image_basename in label_basenames:
                # 有对应标签，保留
                kept_count += 1
            else:
                # 没有对应标签，删除图片
                os.remove(image_file)
                deleted_count += 1
                print(f"已删除: {os.path.basename(image_file)} (无对应标签)")

    # 打印统计结果
    print(f"\n========== 清理完成 ==========")
    print(f"保留的图片数量: {kept_count}")
    print(f"删除的图片数量: {deleted_count}")
    print(f"图片总数: {kept_count + deleted_count}")

if __name__ == "__main__":
    # 设置你的目录路径
    images_directory = "extracted_frames"  # 修改为你的图片目录路径
    labels_directory = "labels"  # 修改为你的标签目录路径

    # 安全确认
    print("警告：此操作将永久删除没有对应标签的图片文件！")
    confirm = input("确认执行？(输入 'yes' 继续): ")

    if confirm.lower() == 'yes':
        # 确保目录存在
        if not os.path.exists(images_directory):
            print(f"错误：图片目录 '{images_directory}' 不存在！")
        elif not os.path.exists(labels_directory):
            print(f"错误：标签目录 '{labels_directory}' 不存在！")
        else:
            delete_images_without_labels(images_directory, labels_directory)
    else:
        print("操作已取消。")