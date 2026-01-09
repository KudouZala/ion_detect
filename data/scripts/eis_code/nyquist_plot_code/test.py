import os
import sys

# 自动修复路径的函数
def fix_path(path):
    # 判断操作系统类型
    if sys.platform == "win32":  # 如果是 Windows 系统
        # 将所有正斜杠替换为反斜杠
        return path.replace("/", "\\")
    elif sys.platform == "linux" or sys.platform == "linux2":  # 如果是 Linux 系统
        # 将所有反斜杠替换为正斜杠
        return path.replace("\\", "/")
    else:
        # 如果是其他操作系统（例如 MacOS），也使用正斜杠
        return path.replace("\\", "/")

# 处理 file_specifications_zhiyun 的函数
def process_file_specifications(file_specifications):
    file_specifications_new = []

    for item in file_specifications:
        # 修复路径
        fixed_path = fix_path(item[0])
        # 将修复后的路径添加到新的列表中
        file_specifications_new.append((fixed_path, item[1], item[2], item[3], item[4]))

    return file_specifications_new

# 示例输入：file_specifications_zhiyun
file_specifications_zhiyun = [
    (r"\home\cagalii\Application\autoeis\AutoEIS\examples\校内测试\20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组1(工步组)(1／1)_工步3(阻抗).csv", (0, 0, 0), '+', '0.1ppm Ca2+ 0h_1107_firecloud', 1),
    (r"\home\cagalii\Application\autoeis\AutoEIS\examples\校内测试\20241107_0.1ppm钙离子污染及恢复测试\新版电解槽_firecloud\20241107_ion\循环1／1_工步组1(工步组)(1／80)_工步3(阻抗).csv", (0, 0, 0), 's', '0.1ppm Ca2+ 2h_1107_firecloud', 1),
]

# 调用函数处理 file_specifications_zhiyun
file_specifications_zhiyun_new = process_file_specifications(file_specifications_zhiyun)

# 打印结果
print("file_specifications_zhiyun_new:", file_specifications_zhiyun_new)



