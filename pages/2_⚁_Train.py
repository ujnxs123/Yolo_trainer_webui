import streamlit as st
import os

from pages.utils.decorator import with_progress_bar
from pages.utils.train_utils import get_gpu_names,save_config_to_toml,find_pt_files,find_pt_files1,load_config_to_session_state,load_global_config,start_train,reset_user_params_to_default,restore_train,upload_file_to_relative_path,create_project_directory



# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# gpu所有id与name


#%%
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 调整结构,三列,多行,布局清晰
# app

st.set_page_config(page_title='模型训练',layout='wide',page_icon='res\\favicon (2).ico')
# 判断读数据
# if

# 调用 model.train() 方法，并传递配置参数

# st.markdown(
#     """
#     <style>
#     [data-testid="stVerticalBlock"]
#     .train{
#         border: 2px solid black;
#         padding: 20px;
#         margin-bottom: 20px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

config_data = load_config_to_session_state("config\yolo_user.toml")





# local_css("./page/css/style.css")
pt_file = st.sidebar.text_input("Upload .pt File",value='/path/to/xxx.pt',help='训练意外中断可上传检查点pt文件,可以以此节点继续训练或者重新开始训练')
col1, col2 = st.columns([2, 1])

with col1:
    st.title("模型训练配置")



    gpu_names = get_gpu_names()
    #  可以自动读取且可多选

    # 设备选择
    
    with st.container(border=True):
        
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
        # data_col1, data_col2 = st.columns(2,gap='small')  
        # # 可能还是上传文件比较好。后期再改
        # with data_col2:
            uploaded_file = st.file_uploader("上传训练数据文件",type='yaml')
        # with data_col1:
            if uploaded_file is not None:
            
                data = st.text_input('训练数据目录(上传)',value= upload_file_to_relative_path(uploaded_file, relative_folder='./datasets/yaml'))
            else:
                data = st.text_input('训练数据目录(默认)',value=config_data['data'])




    # 数据集图片尺寸
    with st.container():
        imgsz = st.number_input("图像尺寸", min_value=512,value=config_data["imgsz"])



    st.subheader('保存设置')
    # 输出设置

    with st.container():
        name = st.text_input("输出名称", config_data["name"])
    with st.container():
        project = st.text_input("输出目录", config_data["project"])
    with st.container():
        save_period = st.number_input("每 N epoch(轮)自动保存一次模型", min_value=-1, value=config_data["save_period"] ,help='设为-1则默认关闭此功能')
    with st.container():
        exist_ok = st.toggle(" 是否覆盖现有的实验" , config_data["exist_ok"] ,help="如果设置为 True,当实验名称已经存在时，将会覆盖现有实验")



    with st.container():
        st.subheader('训练相关参数')
        # epoch设置
        batch = st.number_input("训练batch批次大小", min_value=-1, value=config_data['batch'],help="如果设置为**-1**，则会自动调整批次大小，至你的显卡能容纳的最多图像数量。")
        epochs = st.number_input("最大训练epoch数", min_value=1, value=config_data['epochs'])

    with st.container():
        
        patience = st.number_input("早停等待轮数", min_value=0, value=config_data['patience'],help="启用早停等待功能,在训练过程中，如果在一定的轮数内没有观察到模型性能的明显提升，就会停止训练。可有效防止过拟合")





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
        workers = st.number_input("数据加载时的工作线程数", min_value=0, value=config_data['workers'],help="windows系统默认为0")
        nbs = st.number_input('标准批次大小',min_value=0,value=config_data['nbs'])
        
        lr0	 = st.number_input("初始学习率", value=config_data['lr0'],help='SGD=1E-2, Adam=1E-3 .调整这个值对优化过程至关重要，会影响模型权重的更新速度。')
        lrf = st.number_input("最终学习率", value=config_data['lrf'])          
        momentum = st.number_input('动量优化器',min_value=0.0,value=config_data['momentum'])
        label_smoothing = st.slider('应用标签平滑',min_value=0.0,value=config_data['label_smoothing'],help='标签平滑是一种正则化技术，用于减少模型对训练数据的过拟合程度。') 

        overlap_mask = st.toggle("分割掩码是否应该重叠", value=config_data['overlap_mask'],help='仅用于实例分割任务')
        mask_ratio = st.number_input('蒙版下采样比例',min_value=0,value=config_data["mask_ratio"])
        dropout = st.number_input('使用丢弃正则化',value=config_data["dropout"],help='仅用于分类训练）。如果设置为非零值，则在训练过程中使用丢弃正则化来减少模型的过拟合。')
        optimizer = st.selectbox("优化器类型", ["auto","AdamW", "SGD",'RMSProp','Adam','NAdam','RAdam'])
        weight_decay = st.number_input('优化器的权重衰减',min_value=0.0,value=config_data["weight_decay"],help='权重衰减是一种正则化技术，用于减小模型复杂度，防止过拟合。')

        

        close_mosaic = st.number_input("禁用mosaic增强轮数",min_value=0,value=config_data['close_mosaic'])
        amp = st.toggle("是否使用自动混合精度",value=config_data['amp'],help='利用半精度浮点数加速训练过程，可以减少显存占用。')
        
        cache = st.toggle("缓存潜在特征", value=config_data['cache'])

    

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







# 判断读数据
# if




# yolov8设置
    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("保存配置"):
        # 调用函数写入配置参数
            with st.spinner('Saving...'):
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
            with st.spinner('Resetting...'):
                reset_user_params_to_default('config/yolo_default.toml', 'config/yolo_user.toml')
            st.success("配置参数已成功保存")




    col5,col6 = st.columns([1, 1])
    with col5:

        if 'clicked' not in st.session_state:
            st.session_state.clicked = False


        if st.button("开始训练",help='训练参数从配置文件中获得'):

            create_project_directory(config_data['project'])
            with st.spinner('训练中'):
                result = start_train(config_data)

    with col6:
        if st.button("恢复训练", key="restore_train_button",help='从检查点开始'):
            if isinstance(pt_file, str):
                if os.path.exists(pt_file) and pt_file.endswith('.pt'):
                    with st.spinner('训练中'):
                        restore_train(pt_file)
                else:
                    st.warning(".pt 文件未找到或格式不正确")
            else:
                st.warning("无效的文件路径")

    # 恢复训练以及重新训练代码




            

    
        
    



