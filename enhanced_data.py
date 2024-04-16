import toml
import munch
def load_config(filepath: str):
    # 加载 TOML 文件
    config_dict = toml.load(filepath)
    
    # 使用 Munch 将字典转换为对象
    config = munch.munchify(config_dict)
    
    return config



print( load_config('config\yolo_default.toml'))