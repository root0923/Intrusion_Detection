import os

label_dir = r"data\dataset\with_neg\labels\val"  # 改成你的标签目录

for file_name in os.listdir(label_dir):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(label_dir, file_name)

    with open(file_path, "r") as f:
        lines = f.readlines()

    # 过滤掉 class_id == 2 的行
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        class_id = parts[0]

        if class_id != "2":
            new_lines.append(line)

    # 覆盖写回
    with open(file_path, "w") as f:
        f.writelines(new_lines)

print("处理完成！")