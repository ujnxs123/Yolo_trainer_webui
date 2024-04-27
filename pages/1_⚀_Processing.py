import streamlit as st
import os
from pages.utils.processing_utils import convert_png_to_jpg,split_dataset,modify_yolo_txt_files,convert_dataset_to_yolo,remove_extra_files,process_files,split_datasets,convert_folder_to_yolo,extract_frames
import os
#%%

#----------------------------------------------------------------------------------------------
# app
st.set_page_config(page_title='数据预处理',page_icon='res\\favicon (1).ico') 

st.title('数据集预处理脚本调用页面')

st.write("这个工具可以帮助您对图像数据进行预处理,以准备用于YOLO模型的训练。")



# 数据集是否只有视频形式
video_dataset_enabled = st.checkbox("视频数据集处理", value=False)

# 如果开启了视频数据集功能，则显示上传视频和输入目标地址功能
if video_dataset_enabled:
    uploaded_video = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])
    frame_interval = st.slider("抽帧间隔", min_value=1, max_value=100, value=10)
    target_folder = st.text_input("目标地址", "./datasets/Temporary_data")
    if st.button("执行视频帧提取"):
            if uploaded_video is not None:
                # 调用提取帧的函数
                with st.spinner('Running'):
                    extract_frames(uploaded_video, target_folder, frame_interval)
                st.success("视频帧提取完成！")
            else:
                st.warning("请先上传视频文件。")

#%%
# functions = {
#     "图片格式转换": convert_png_to_jpg,
#     "检测split": split_dataset,
#     "改变类别编号": modify_yolo_txt_files,
#     "两点voc格式转yolo格式": convert_dataset_to_yolo,
#     "删除不对应的标注和图片": remove_extra_files,
#     "四点voc格式转yolo格式": process_files,
#     "分类split": split_datasets,
#     "一对一coco转yolo": convert_folder_to_yolo
# }

# # 显示选择框
# selected_function = st.selectbox('选择要运行的脚本', list(functions.keys()))

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(["图片格式转换", "检测split","改变类别编号","两点voc格式转yolo格式","删除不对应的标注和图片","四点voc格式转yolo格式","分类split","一对一coco转yolo"])


# if selected_function == "图片格式转换":
with tab1:
    st.subheader("转换 PNG 到 JPG")
    output_folder = st.text_input('output_folder', value='/path/to/output_folder',key='1')
    if st.button("运行",key='5'):
        result = convert_png_to_jpg( output_folder)
        

# elif selected_function == "检测split":
with tab2:
    st.write("## 分割数据集")
    image_folder = st.text_input('image_folder', value='/path/to/image_folder',key='11')
    label_folder = st.text_input('label_folder', value='/path/to/label_folder')
    output_folder = st.text_input('output_folder',value= '/path/to/output_folder',key='2')
    train_ratio = st.slider("训练集比例", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    val_ratio = st.slider("验证集比例", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    test_ratio = st.slider("测试集比例", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    if st.button("运行",key='6'):
        result = split_dataset(image_folder,label_folder, output_folder,  train_ratio, val_ratio, test_ratio)
        

# elif selected_function == "改变类别编号":
with tab3:
    st.write("## 修改 YOLO 文本文件")
    input_folder = st.text_input("输入文件夹路径", value="./input_folder")
    target_category = st.text_input("输出文件夹路径", value="./output_folder")
    class_names = st.text_input("类别名称 (用逗号分隔)", value="cat, dog")
    if st.button("运行",key='7'):
        result = modify_yolo_txt_files(input_folder, target_category, class_names.split(","))
        st.write(result)


# elif selected_function == "两点voc格式转yolo格式":
with tab4:
    st.write("## 将数据集转换为 YOLO 格式")
    input_folder = st.text_input("输入xml文件夹路径",value= "./input_folder")
    image_folder = st.text_input('image_folder',value= '/path/to/image_folder',key='12')
    output_folder = st.text_input('output_folder', value='/path/to/output_folder',key='3')
    if st.button("运行",key='8'):
        if input_folder and image_folder and output_folder:
            convert_dataset_to_yolo(input_folder, image_folder, output_folder)
    else:
        st.warning("请填写所有必要的路径")
           

# elif selected_function == "删除不对应的标注和图片":
with tab5:
    st.write("## 删除额外文件")
    jpg_folder_path = st.text_input("请输入 JPG 文件夹路径：")
    xml_folder_path = st.text_input("请输入 XML 文件夹路径：")
    if st.button("运行",key='9'):
        if jpg_folder_path and xml_folder_path:
            remove_extra_files(jpg_folder_path, xml_folder_path)
    else:
        st.warning("请填写所有必要的路径")
        

# elif selected_function == "四点voc格式转yolo格式":
with tab6:
    st.write("## 处理文件")
    postfix = st.text_input("请输入后缀：")
    imgpath = st.text_input("请输入图片文件夹路径：")
    xmlpath = st.text_input("请输入XML文件夹路径：")
    txtpath = st.text_input("请输入输出 TXT 文件夹路径：")
    if st.button("运行",key='10'):
        if postfix and imgpath and xmlpath and txtpath:
            process_files(postfix, imgpath, xmlpath, txtpath)
    else:
        st.warning("请填写所有必要的参数")

# elif selected_function == "分类split":
with tab7:
    st.write("## 分割数据集")
    origin_path = st.text_input("请输入原始数据集路径：")
    train_ratio = st.slider("训练集比", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    val_ratio = st.slider("验证集比", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    test_ratio = st.slider("测试集比", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    if st.button("分割数据集"):
        if origin_path:
            split_datasets(origin_path, train_ratio, val_ratio, test_ratio)
        else:
            st.warning("请输入原始数据集路径")

# elif selected_function == "一对一coco转yolo":
with tab8:
    st.write("## 转换为 YOLO 格式")
    json_folder = st.text_input("请输入包含 JSON 文件的文件夹路径：")
    if st.button("转换为 YOLO 格式"):
        if json_folder:
            convert_folder_to_yolo(json_folder)
        else:
            st.warning("请填写 JSON 文件夹路径")
    

# 上传文件

# ----------------------------------------------------------------------------

# --------------------------------------------------------------------------


#%%
