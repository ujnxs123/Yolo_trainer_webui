import streamlit as st
import requests
import os
from PIL import Image
import cv2
from pathlib import Path

# 对单个图片resize
def letterbox(img, new_shape, color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img

def batch_resize(project_name, uploaded_folder, new_shape):
    output_folder = Path(f"./datasets/{project_name}/images")
    output_folder.mkdir(parents=True, exist_ok=True)

    # 遍历上传的图像文件夹中的所有图像文件
    for filename in os.listdir(uploaded_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 仅处理图片文件
            # 读取图像
            img = cv2.imread(os.path.join(uploaded_folder, filename))
            # 对图像应用letterbox方法进行resize
            img_resized = letterbox(img, new_shape=new_shape)
            # 保存处理后的图像到指定目录
            cv2.imwrite(str(output_folder / filename), img_resized)

    return output_folder


def preprocess_data(image_folder, label_names):
    pass


# .................................................................
def extract_frames(video_path, output_folder, frame_interval):
    
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        # 读取视频帧
        ret, frame = video_capture.read()

        # 如果没有成功读取帧，说明到达视频的结尾
        if not ret:
            break

        # 每隔指定的帧间隔保存一帧图像
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    # 释放视频捕获对象
    video_capture.release()




def app():
    st.title("YOLO数据预处理")
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
                    extract_frames(uploaded_video.name, target_folder, frame_interval)
                    st.success("视频帧提取完成！")
                else:
                    st.warning("请先上传视频文件。")


    # 上传文件
    uploaded_folder = st.file_uploader("上传图像文件夹", type="folder")


    target_width = st.number_input("目标宽度", value=640)
    target_height = st.number_input("目标高度", value=640)
    new_shape = (target_width, target_height)
    project_name =st.text_input("项目名称",help="会在data下创建同名的文件夹")
# ----------------------------------------------------------------------------
#  接下来需要做得： 根据target 进行resize保存到dataset里, 再是数据增强保存在项目名称下的dataset里。再然后是引用那个打标软件，进行打标，然后是分别分割成train，test，val
# --------------------------------------------------------------------------
    if st.button("Resize"):
        if not project_name or not uploaded_folder:
            st.warning("项目名称和上传的图像文件夹不能为空！")
            return
        
        # try except 方法可以放在函数体里？会比较好
        try:
            output_folder = batch_resize(project_name, uploaded_folder, new_shape)
            st.success(f"预处理完成！处理后的图像已保存在 {output_folder} 中。")
        except:
            st.error("wrong")



    data_augmentation_enabled = st.checkbox("是否应用数据增强", value=False, help="应用数据增强可增加数据的多样性，有助于提高模型的泛化能力")
    if data_augmentation_enabled:

        pass




    label_names = st.text_input("类别标签", help="逗号分隔的类别名称,例如,cat,dog")


    if st.button("开始预处理"):
        if uploaded_folder is not None:
            try:
                # 执行数据预处理
                images = os.listdir(uploaded_folder)
                preprocessed_images, labels = preprocess_data(images, label_names)
                
                # 结果展示
                st.write("预处理完成！")
                st.write("预处理后的图像和标签信息：")
                for image, label in zip(preprocessed_images, labels):
                    st.image(image, caption=label, width=200)
            except Exception as e:
                st.error(f"预处理出错：{e}")