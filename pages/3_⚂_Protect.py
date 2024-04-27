from pathlib import Path
from PIL import Image
import streamlit as st
import config as config
import time
import os
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()

     #   cap = cv2.VideoCapture(0)
        # 获取视频帧的维度
        frame_width = int(vid_cap.get(3))
        frame_height = int(vid_cap.get(4))

        # frame_count = 150  # 5秒视频，假设每秒30帧
        frame_rate = 30

        frames = []
        # 创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out1 = cv2.VideoWriter(r'.\\outputs\\Total\\new.mp4', fourcc, 20.0, (frame_width, frame_height))
        # 循环视频帧
        save_flag = False  # 保存视频的标志
        elapsed_time = 0  # 初始时间为0
        target_class_value = 0  # 假设目标类别值为1
        start_time = time.time()


        while not flag:
            success, frame = vid_cap.read()

            if success:
                _display_detected_frames(
                            conf,
                            model,
                            st_frame,
                            frame
                        )
                frames.append(frame)
                # 使用yolov8进行预测
                results = model(frame)
                # 可视化结果
                annotated_frame = results[0].plot()
                # 将带注释的帧写入视频文件
                out1.write(annotated_frame)
                for r in results:
                    boxes = r.boxes  # Boxes object for bbox outputs
                    if boxes.cls.numel() != 0:
                        # cls_value = boxes.cls.item()
                        # print('cls是', cls_value)
                        cls_tensor = boxes.cls
                        cls_list = cls_tensor.cpu().numpy().tolist()
                    else:
                        cls_list = [99999]
                        print('boxes.cls 中没有元素，无法转换为标量', cls_list)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print('间隔时间', elapsed_time)
                    if elapsed_time > 10 and save_flag:  # 重置保存标志
                        save_flag = False
                    if target_class_value in cls_list and not save_flag:
                        # if target_class_value in cls_list:
                        timestamp = int(time.time())
                        img_name = os.path.join('./outputs/Screenshot',f'{timestamp}.jpg')
                        cv2.imwrite(img_name, frame)
                        print(f'保存图片: {img_name}')

                        frame_count = len(frames)
                        print('帧数', frame_count)

                        start_frame = max(0, frame_count - frame_rate * 10)  # 判断起始帧

                        end_frame = min(frame_count + frame_rate * 5, len(frames))  # 判断结束帧
                        video_name = os.path.join('./outputs/Screenshot' , f'video_{timestamp}.avi')
                        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate,
                                              (frame.shape[1], frame.shape[0]))
                        for i in range(start_frame, end_frame):
                            out.write(frames[i])
                            print(i)
                        out.release()
                        print(f'保存视频: {video_name}')

                    
                        save_flag = True  # 设置保存标志为 True
                        frames = frames[start_frame:end_frame]
                        start_time = time.time()

            else:
                # 最后结尾中断视频帧循环
                vid_cap.release()
                break
            # success, image = vid_cap.read()
            # if success:
            #     _display_detected_frames(
            #         conf,
            #         model,
            #         st_frame,
            #         image
            #     )
            # else:
            #     vid_cap.release()
            #     break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")


# ----------------------------------------------------------------------------------------------------'
# app
st.set_page_config(
    page_title="智慧工地检测系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )
st.title('智慧工地目标检测系统')

st.sidebar.header("模型")
# sidebar
st.sidebar.header("模型配置选择")

# model options
task_type = st.sidebar.selectbox(
    "任务类别选择",
    ["检测", "分割", "关键点",'Outputs'],
)

model_type = None
if task_type == "检测":
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.DETECTION_MODEL_LIST
    )
elif task_type == "分割":
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.SEGMENT_MODEL_LIST
    )
elif task_type == "关键点":
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.POSE_MODEL_LIST
    )
elif task_type == "Outputs":
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.OUTPUTS_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "置信度", 30, 100, 50)) / 100

model_path = ""
if model_type:

    if task_type == "检测":
        model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    elif task_type == "分割":
        model_path = Path(config.SEGMENT_MODEL_DIR, str(model_type))
    elif task_type == "关键点":
        model_path = Path(config.POSE_MODEL_DIR, str(model_type))
    elif task_type == "Outputs":
        model_path = Path(config.OUTPUTS_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "选择来源",
    config.SOURCES_LIST
)


source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")

# st.button('提交申述')