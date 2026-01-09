def get_ion_color(ion_info: str) -> tuple:
    # 定义离子颜色方案
    ion_colors = { 
    "钠离子": {
        "2ppm": (255, 230, 100),  # 浅黄色，偏黄
        "10ppm": (255, 120, 50),   # 中等橙色
    },
    "钙离子": {
        "2ppm": (200, 150, 255),  # 浅紫色
        "10ppm": (140, 60, 255),   # 中等紫色
    },
    "铬离子": {
        "2ppm": (150, 255, 150),  # 浅绿色
        "10ppm": (0, 255, 0),     # 鲜绿色
    },
    "镍离子": {
        "2ppm": (102, 255, 178),  # 淡青色
        "10ppm": (0, 204, 128),   # 明亮青色
    },
    "铜离子": {
        "2ppm": (180, 240, 255),  # 浅蓝色
        "10ppm": (0, 128, 255),   # 标准蓝色
    },
    "铁离子": {
        "2ppm": (255, 180, 80),   # 淡橙色
        "10ppm": (255, 100, 0),    # 深橙色
    },
    "铝离子": {
        "2ppm": (230, 240, 255),  # 浅青蓝色
        "10ppm": (100, 180, 255), # 中等青蓝色
    },
    "无离子": {
        "default": (200, 200, 200)  # 灰色
    },
    "恢复后": {
        "default": (150, 150, 150)  # 恢复后的统一灰色
    }
    }   

    # 解析输入信息，提取离子类型和浓度
    ion_type = ""
    concentration = ""
    if "钠离子" in ion_info:
        ion_type = "钠离子"
    elif "钙离子" in ion_info:
        ion_type = "钙离子"
    elif "铬离子" in ion_info:
        ion_type = "铬离子"
    elif "镍离子" in ion_info:
        ion_type = "镍离子"
    elif "铜离子" in ion_info:
        ion_type = "铜离子"
    elif "铁离子" in ion_info:
        ion_type = "铁离子"
    elif "铝离子" in ion_info:
        ion_type = "铝离子"
    elif "无离子" in ion_info:
        ion_type = "无离子"
    elif "恢复后" in ion_info:
        ion_type = "恢复后"

    if "2ppm" in ion_info:
        concentration = "2ppm"
    elif "10ppm" in ion_info:
        concentration = "10ppm"
    elif "100ppm" in ion_info:
        concentration = "100ppm"
    else:
        concentration = "default"

    # 获取对应的RGB值
    color = ion_colors.get(ion_type, {}).get(concentration, (0, 0, 0))  # 默认返回黑色
    return color

# 示例用法
ion_info = "钙离子 10ppm"
rgb_value = get_ion_color(ion_info)
print(f"颜色RGB值为: {rgb_value}")