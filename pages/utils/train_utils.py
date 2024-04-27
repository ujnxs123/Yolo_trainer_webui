import streamlit as st
import os
import subprocess
from ultralytics import YOLO
import traceback
import toml
import munch

#%%
# gpu所有id与name
def get_gpu_names():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
        gpu_names = [name.strip() for name in result.stdout.strip().split('\n')]
        return gpu_names
    except Exception as e:
        st.error("Error occurred:", e)
        return []


#%%
# 保存至toml中

def save_config_to_toml(device, data, train_model, imgsz, name, project, save_period, batch, epochs, patience, exist_ok, workers, nbs, lr0, lrf, momentum, weight_decay, close_mosaic, label_smoothing, overlap_mask, mask_ratio, dropout, optimizer, amp, cache, toml_file):
    config = {
        'device': {'device': device},
        'data': {'data': data},
        'train': {
            'model': train_model,
            'imgsz': imgsz,
            'name': name,
            'project': project,
            'save_period': save_period,
            'batch': batch,
            'epochs': epochs,
            'patience': patience,
            'exist_ok': exist_ok
        },
        'advanced': {
            'workers': workers,
            'nbs': nbs,
            'lr0': lr0,
            'lrf': lrf,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'close_mosaic': close_mosaic,
            'label_smoothing': label_smoothing,
            'overlap_mask': overlap_mask,
            'mask_ratio': mask_ratio,
            'dropout': dropout,
            'optimizer': optimizer,
            'amp': amp,
            'cache': cache
        }
        
    }
    with open(toml_file, 'w') as f:
        toml.dump(config, f)




#%%
# 获得目标文件夹下所有pt后缀的
def find_pt_files(folder_path):
    pt_files = []
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pt") or file.endswith(".yaml"):
                    pt_files.append(os.path.join(root, file))
    except Exception as e:
        st.error("Error occurred:",e)
    return pt_files
# 获得目标文件夹下所有yaml后缀的
def find_pt_files1(folder_path):
    pt_files = []
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if  file.endswith(".yaml"):
                    pt_files.append(os.path.join(root, file))
    except Exception as e:
        st.error("Error occurred:",e)
    return pt_files

# seesion_state作为全局变量传递函数，如果执行完才设为flase，执行完按钮才生效

#%%
# 调用该函数则上传yolo参数为全局变量
# 只有第一次修改load，第二次第三次修改使用的全局变量还是第一次修改的
def load_config_to_session_state(toml_path):
    # 加载配置数据
    config = load_global_config(toml_path)
    
    # 存储配置数据到 st.session_state
    if 'config' not in st.session_state:
        st.session_state.config = {}

    st.session_state.config = {
        "train_model": config.train.model,
        "data": config.data.data,
        "imgsz": config.train.imgsz,
        "name": config.train.name,
        "project": config.train.project,
        "save_period": config.train.save_period,
        "batch": config.train.batch,
        "epochs": config.train.epochs,
        "patience": config.train.patience,
        "exist_ok": config.train.exist_ok,
        "device": config.device.device,
        "workers": config.advanced.workers,
        "nbs": config.advanced.nbs,
        "lr0": config.advanced.lr0,
        "lrf": config.advanced.lrf,
        "momentum": config.advanced.momentum,
        "weight_decay": config.advanced.weight_decay,
        "close_mosaic": config.advanced.close_mosaic,
        "label_smoothing": config.advanced.label_smoothing,
        "overlap_mask": config.advanced.overlap_mask,
        "mask_ratio": config.advanced.mask_ratio,
        "dropout": config.advanced.dropout,
        "optimizer": config.advanced.optimizer,
        "amp": config.advanced.amp,
        "cache": config.advanced.cache
    }
    return st.session_state.config

# yolov8开始训练
#%%
def start_train(config_data):
    # 数据导入

    
    train_model= config_data['train_model']

    if st.session_state.clicked:
        st.warning("训练正在进行中，请等待当前训练完成。")
        return

    # 将按钮点击状态设置为已点击

    try:
        # 执行训练任务
        st.session_state.clicked = True
        model = YOLO(train_model)
        result = model.train(
            data=config_data["data"],
            imgsz=config_data["imgsz"],
            name=config_data["name"],
            project=config_data["project"],
            save_period=config_data["save_period"],
            batch=config_data["batch"],
            epochs=config_data["epochs"],
            patience=config_data["patience"],
            exist_ok=config_data["exist_ok"],
            device=config_data["device"],
            workers=config_data["workers"],
            nbs=config_data["nbs"],
            lr0=config_data["lr0"],
            lrf=config_data["lrf"],
            momentum=config_data["momentum"],
            weight_decay=config_data["weight_decay"],
            close_mosaic=config_data["close_mosaic"],
            label_smoothing=config_data["label_smoothing"],
            overlap_mask=config_data["overlap_mask"],
            mask_ratio=config_data["mask_ratio"],
            dropout=config_data["dropout"],
            optimizer=config_data["optimizer"],
            amp=config_data["amp"],
            cache=config_data["cache"]
        )
        st.success('训练成功')
        return result
    except Exception as e:
        traceback.print_exc()
        st.error('训练出错：{}'.format(str(e)))
        
    finally:
        # 训练完成后，将程序状态设置为未执行
        st.session_state.clicked = False


    
        
# 恢复训练以及重新开始训    
#%%
def restore_train(file_path):
    if st.session_state.clicked:
        st.warning("训练正在进行中，请等待当前训练完成。")
        return

    # 将按钮点击状态设置为已点击
   
    try:
        st.session_state.clicked = True
        if file_path is not None:
           
            # 如果找到最新的 .pt 文件，则加载部分训练好的模型
            model = YOLO(file_path)
            # 继续训练
            results = model.train(resume=True)  # 恢复训练
            st.success('恢复训练成功')
            return results
        else:
            st.warning('上传文件失败')

    except Exception as e:
        traceback.print_exc()
        st.error('恢复训练出错：{}'.format(str(e)))
    finally:
    # 训练完成后，将程序状态设置为未执行
        st.session_state.clicked = False
   



#%%
# 读取配置文件
def load_global_config( filepath : str ):
    config_dict = toml.load(filepath)
    
    # 使用 Munch 将字典转换为对象
    config = munch.munchify(config_dict)
    
    return config


# 重置参数功能
#%%
def reset_user_params_to_default(default_file, user_file):
    # 读取默认参数文件
    with open(default_file, 'r') as f:
        default_params = toml.load(f)
    
    # 读取用户参数文件
    with open(user_file, 'r') as f:
        user_params = toml.load(f)
    
    # 将用户参数文件中的值替换为默认参数文件中的值
    for section, values in default_params.items():
        for key, value in values.items():
            if section in user_params and key in user_params[section]:
                user_params[section][key] = value
    
    # 将修改后的参数保存回用户参数文件中
    with open(user_file, 'w') as f:
        toml.dump(user_params, f)


#%%
# 上传文件到相对位置并返回其位置
def upload_file_to_relative_path(uploaded_file, relative_folder):
    try:
        # 检查上传文件是否为空
        if uploaded_file is None:
            st.error("未选择文件")
            return None
        
        base_dir = os.getcwd()
        
        # 构建相对路径的完整路径
        relative_path = os.path.join(base_dir, relative_folder)
        
        # 检查相对路径是否存在，如果不存在则创建
        if not os.path.exists(relative_path):
            os.makedirs(relative_path)

        # 获取上传文件的文件名
        file_name = uploaded_file.name
        
        # 构建上传文件的完整路径
        relative_file_path = os.path.join(relative_path, file_name)
        
        # 将上传文件保存到指定相对路径上
        with open(relative_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.success('上传成功')
        return os.path.abspath(relative_file_path)

       
    except Exception as e:
        st.error(f'上传失败: {str(e)}')
        return None
    
#%%
# 不存在则创建对应文件夹
def create_project_directory(project_path):
    if not os.path.exists(project_path):
        os.makedirs(project_path)