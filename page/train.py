import streamlit as st
import os
import subprocess
from ultralytics import YOLO
import traceback
import toml
import munch
from functools import wraps
import time
def retry(num_retries, exception_to_check, sleep_time=0):
    """
    Decorator to retry a function in case of specific exceptions.
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(1, num_retries+1):
                try:
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    st.warning(f"{func.__name__} raised {e.__class__.__name__}. Retrying...")
                    if i < num_retries:
                        time.sleep(sleep_time)
            # Raise the exception if still not successful after multiple retries
            raise e
        return wrapper
    return decorate


def click_button():
    st.session_state.clicked = True

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# gpu所有id与name
def get_gpu_names():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
        gpu_names = [name.strip() for name in result.stdout.strip().split('\n')]
        return gpu_names
    except Exception as e:
        st.error("Error occurred:", e)
        return []



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




# 获得目标文件夹最新的pt文件以继续训练或者重新开始训练
def get_latest_pt_file(folder_path, name):
    latest_pt_file = None
    latest_time = 0
    
    # 遍历当前文件夹下所有条目
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name == name:
                folder_path = entry.path  # 更新文件夹路径
                break
        
    # 在更新后的文件夹路径中搜索最新的 .pt 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_pt_file = file_path
    
    return latest_pt_file


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
# yolov8开始训练
def start_train(toml_path):
    # 数据导入

    config = load_global_config(toml_path)
    train_model = config.train.model
    data = config.data.data
    imgsz = config.train.imgsz
    name = config.train.name
    project = config.train.project
    save_period = config.train.save_period
    batch = config.train.batch
    epochs = config.train.epochs
    patience = config.train.patience
    exist_ok = config.train.exist_ok
    device = config .device.device
    # 获取高级参数
    workers = config.advanced.workers
    nbs = config.advanced.nbs
    lr0 = config.advanced.lr0
    lrf = config.advanced.lrf
    momentum = config.advanced.momentum
    weight_decay = config.advanced.weight_decay
    close_mosaic = config.advanced.close_mosaic
    label_smoothing = config.advanced.label_smoothing
    overlap_mask = config.advanced.overlap_mask
    mask_ratio = config.advanced.mask_ratio
    dropout = config.advanced.dropout
    optimizer = config.advanced.optimizer
    amp = config.advanced.amp
    cache = config.advanced.cache


    if st.session_state.clicked:
        st.warning("训练正在进行中，请等待当前训练完成。")
        return

    # 将按钮点击状态设置为已点击
    click_button()

    try:
        # 执行训练任务
        st.session_state.clicked = True
        model = YOLO(train_model)
        result = model.train(
            data=data,
            imgsz=imgsz,
            name=name,
            project=project,
            save_period=save_period,
            batch=batch,
            epochs=epochs,
            patience=patience,
            exist_ok=exist_ok,
            device=device,
            workers=workers,
            nbs=nbs,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            close_mosaic=close_mosaic,
            label_smoothing=label_smoothing,
            overlap_mask=overlap_mask,
            mask_ratio=mask_ratio,
            dropout=dropout,
            optimizer=optimizer,
            amp=amp,
            cache=cache
        )
       
        return result
    except Exception as e:
        traceback.print_exc()
        st.error('运行出现错误：{}'.format(str(e)))
        
    finally:
        # 训练完成后，将程序状态设置为未执行
        st.session_state.clicked = False
        


def resetart_train(output_dir,output_name):
    latest_pt_file = get_latest_pt_file(output_dir, output_name)
    if latest_pt_file:
        # 如果找到最新的 .pt 文件，则加载部分训练好的模型
        model = YOLO(latest_pt_file)
        # 继续训练
        results = model.train(resume=False)  # 恢复训练
    else:
        st.error("No .pt file found")
    return results

def restore_train(output_dir,output_name):
    latest_pt_file = get_latest_pt_file(output_dir, output_name)
    if latest_pt_file:
        # 如果找到最新的 .pt 文件，则加载部分训练好的模型
        model = YOLO(latest_pt_file)
        # 继续训练
        results = model.train(resume=True)  # 恢复训练
    else:
        st.error("No .pt file found")

    return results


# 读取配置文件
def load_global_config( filepath : str ):
    config_dict = toml.load(filepath)
    
    # 使用 Munch 将字典转换为对象
    config = munch.munchify(config_dict)
    
    return config


# 重置参数功能
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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 调整结构,三列,多行,布局清晰
def app():
    # local_css("./page/css/style.css")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("模型训练配置")



        gpu_names = get_gpu_names()
        #  可以自动读取且可多选

        # 设备选择
        with st.container():
            selected_gpus = st.multiselect("训练设备", gpu_names, default=gpu_names,help="不选择任何设备则默认为使用cpu训练,可多显卡训练")
            if not selected_gpus:  # 如果没有选择GPU，则默认选择CPU
                device = "cpu"
            else:
                device = [gpu_names.index(gpu_name) for gpu_name in selected_gpus]


        st.subheader('训练用模型')
        with st.container():
            model = st.selectbox("底模文件路径", find_pt_files('./weights'),help="选择底模文件以进行训练。")




        st.subheader('数据集设置')
        # 数据集目录fddc
        with st.container():
            # 可能还是上传文件比较好。后期再改
            data = st.selectbox("训练数据目录", find_pt_files1('./'))
        # 数据集图片尺寸
        with st.container():
            imgsz = st.number_input("图像尺寸", min_value=512,value=640)



        st.subheader('保存设置')
        # 输出设置

        with st.container():
            name = st.text_input("输出名称", "demo")
        with st.container():
            project = st.text_input("输出目录", "./outputs")
        with st.container():
            save_period = st.number_input("每 N epoch(轮)自动保存一次模型", min_value=-1, value=10,help='设为-1则默认关闭此功能')
        with st.container():
            exist_ok = st.checkbox(" 是否覆盖现有的实验" , value=False ,help="如果设置为 True,当实验名称已经存在时，将会覆盖现有实验")



        with st.container():
            st.subheader('训练相关参数')
            # epoch设置
            batch = st.number_input("训练batch批次大小", min_value=-1, value=-1,help="如果设置为**-1**，则会自动调整批次大小，至你的显卡能容纳的最多图像数量。")
            epochs = st.number_input("最大训练epoch数", min_value=1, value=100)
        
        with st.container():
            
            patience = st.number_input("早停等待轮数", min_value=0, value=100,help="启用早停等待功能,在训练过程中，如果在一定的轮数内没有观察到模型性能的明显提升，就会停止训练。可有效防止过拟合")


        


    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 后期再配置，先写基本配置就能跑的

        # 用这个方式训练模型就要写两套了
        # with st.container():
            # # 添加开关控件，用于控制是否启用专家配置模式，默认关闭
            # expert_mode_enabled = st.checkbox("启用专家配置模式",value=False)

            # # 如果启用了专家配置模式，则显示更多的配置选项
            # if expert_mode_enabled:

# 展开类型 
        with st.expander("高级配置"):
            workers = st.number_input("数据加载时的工作线程数", min_value=0, value=0,help="windows系统默认为0")
            nbs = st.number_input('标准批次大小',min_value=0,value=64)
            
            lr0	 = st.number_input("初始学习率", value=0.01,help='SGD=1E-2, Adam=1E-3 .调整这个值对优化过程至关重要，会影响模型权重的更新速度。')
            lrf = st.number_input("最终学习率", value=0.01)          
            momentum = st.number_input('动量优化器',min_value=0.0,value=0.937)
            label_smoothing = st.slider('应用标签平滑',min_value=0.0,value=0.0,help='标签平滑是一种正则化技术，用于减少模型对训练数据的过拟合程度。') 

            overlap_mask = st.checkbox("分割掩码是否应该重叠", value=True,help='仅用于实例分割任务')
            mask_ratio = st.number_input('蒙版下采样比例',min_value=0,value=4)
            dropout = st.number_input('使用丢弃正则化',value=0.0,help='仅用于分类训练）。如果设置为非零值，则在训练过程中使用丢弃正则化来减少模型的过拟合。')
            optimizer = st.selectbox("优化器类型", ["auto","AdamW", "SGD",'RMSProp','Adam','NAdam','RAdam'])
            weight_decay = st.number_input('优化器的权重衰减',min_value=0.0,value=0.0005,help='权重衰减是一种正则化技术，用于减小模型复杂度，防止过拟合。')

            
           
            close_mosaic = st.number_input("禁用mosaic增强轮数",min_value=0,value=10)
            amp = st.checkbox("是否使用自动混合精度",value=True,help='利用半精度浮点数加速训练过程，可以减少显存占用。')
            
            cache = st.checkbox("缓存潜在特征", value=False)

            

    with col2:
        st.subheader("参数检查")
        st.write(f"device: {device}")
        st.write(f"model: {model}")
        st.write(f'data:{data}')
        st.write(f"imgsz: {imgsz}")
        st.write(f"name: {name}")
        st.write(f"project: {project}")
        st.write(f"save_period: {save_period}")
        st.write(f"batch: {batch}")
        st.write(f"epochs: {epochs}")
        st.write(f"patience: {patience}")
        st.write(f"exist_ok: {exist_ok}")
        st.write(f"workers: {workers}")
        st.write(f"nbs: {nbs}")
        st.write(f"lr0: {lr0}")
        st.write(f"lrf: {lrf}")
        st.write(f"momentum: {momentum}")
        st.write(f"weight_decay: {weight_decay}")
        st.write(f"close_mosaic: {close_mosaic}")
        st.write(f"label_smoothing: {label_smoothing}")
        st.write(f"overlap_mask: {overlap_mask}")
        st.write(f"mask_ratio: {mask_ratio}")
        st.write(f"dropout: {dropout}")
        st.write(f"optimizer: {optimizer}")
        st.write(f"amp: {amp}")
        st.write(f"cache: {cache}")







       

    # yolov8设置
        col3, col4 = st.columns([1, 1])
        with col3:
            if st.button("保存配置"):
            # 调用函数写入配置参数
                save_config_to_toml(device, 
                                    data,
                                    model,
                                      imgsz,
                                        name,
                                          project,
                                            save_period,
                                              batch, 
                                              epochs, patience, exist_ok, workers, nbs,
                                                lr0, lrf, momentum, weight_decay, close_mosaic,
                                                  label_smoothing, overlap_mask, mask_ratio, 
                                                  dropout, optimizer, amp, cache,
                                                    'config\yolo_user.toml')
                st.success("配置参数已成功保存")
        with col4:
            if st.button("重置配置"):
                reset_user_params_to_default('config/yolo_default.toml', 'config/yolo_user.toml')
                st.success("配置参数已成功保存")





        with st.container():
            if 'clicked' not in st.session_state:
                st.session_state.clicked = False
                

            if st.button("开始训练"):
            
                result = start_train('config\yolo_user.toml')
                
            
            
                
            

    

