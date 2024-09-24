# 打开文件并读取所有行
with open("/data/home/scv7454/run/GraduationProject/fix.txt", 'r') as f:
    lines = f.readlines()

# 用于存储上一行的内容
previous_line = None
# 用于存储结果
results = []
# 标记是否已经记录过连续的错误
found_consecutive = False

# 遍历每一行
for i in range(len(lines)):
    if "index 0 is out of bounds for axis 0 with size 0" in lines[i]:
        if not found_consecutive and previous_line is not None:
            results.append(previous_line.strip())  # 记录上一行
            found_consecutive = True  # 标记为已找到连续错误
    else:
        found_consecutive = False  # 重置标记
        previous_line = lines[i]  # 更新上一行

# 输出结果，使用空格作为间隔符
print(" ".join(str(int(float(result) * 1000 / 2)) for result in results))
