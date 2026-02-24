import os

def clean_labels(images_dir, labels_dir):
    # 获取images目录中的所有文件名（不含扩展名）
    image_names = set()
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            name_without_ext = os.path.splitext(filename)[0]
            image_names.add(name_without_ext)
    
    print(f"Found {len(image_names)} image files in images directory")
    
    # 获取labels目录中的所有txt文件
    txt_files = []
    for filename in os.listdir(labels_dir):
        if filename.lower().endswith('.txt'):
            txt_files.append(filename)
    
    print(f"Found {len(txt_files)} txt files in labels directory")
    
    # 找出多余的文件（在images中不存在的）
    extra_files = []
    empty_files = []
    
    for txt_file in txt_files:
        name_without_ext = os.path.splitext(txt_file)[0]
        
        # 检查是否在images中存在对应的文件
        if name_without_ext not in image_names:
            extra_files.append(txt_file)
        
        # 检查文件是否为空
        txt_path = os.path.join(labels_dir, txt_file)
        if os.path.getsize(txt_path) == 0:
            empty_files.append(txt_file)
    
    print(f"Found {len(extra_files)} extra label files")
    print(f"Found {len(empty_files)} empty label files")
    
    # 删除多余的文件
    deleted_count = 0
    for extra_file in extra_files:
        file_path = os.path.join(labels_dir, extra_file)
        os.remove(file_path)
        print(f"Deleted extra file: {extra_file}")
        deleted_count += 1
    
    # 删除空文件
    for empty_file in empty_files:
        if empty_file not in [os.path.basename(f) for f in extra_files]:  # 避免重复删除
            file_path = os.path.join(labels_dir, empty_file)
            os.remove(file_path)
            print(f"Deleted empty file: {empty_file}")
            deleted_count += 1
    
    print(f"Total deleted files: {deleted_count}")

if __name__ == "__main__":
    images_dir = "/home/ysy/object_detection/intrusion/Intrusion_Detection/data/dataset/swimming/images"
    labels_dir = "/home/ysy/object_detection/intrusion/Intrusion_Detection/data/dataset/swimming/labels"
    clean_labels(images_dir, labels_dir)