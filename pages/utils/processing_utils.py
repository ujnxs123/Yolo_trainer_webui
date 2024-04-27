import json
import os
from PIL import Image
import random
import xml.etree.ElementTree as ET
import tempfile
import os
import cv2
import numpy as np
import shutil
import streamlit as st
#%%
# 一对一coco转yolo
def normalize_coordinates(x, y, img_width, img_height):
    try:
        normalized_x = x / img_width
        normalized_y = y / img_height
        return normalized_x, normalized_y
    except ZeroDivisionError:
        st.error("Error: Image width or height cannot be zero.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None

def convert_to_yolo(json_file, output_file=None):
    labels = ["unlabeled", "ego vehicle", "rectification border", "out of roi", "static", "dynamic", "ground", "road", "sidewalk", "parking", "rail track", "building", "wall", "fence", "guard rail", "bridge", "tunnel", "pole", "polegroup", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle", "license plate","cargroup","bicyclegroup",'persongroup']

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: File {json_file} not found.")
        return
    except json.JSONDecodeError:
        st.error(f"Error: Invalid JSON file: {json_file}")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return

    if output_file is None:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_file = f"{base_name}.txt"

    try:
        with open(output_file, 'w') as f:
            for obj in data.get('objects', []):
                label = obj.get('label', '')
                if label in labels:
                    label_index = labels.index(label)

                    f.write(f"{label_index} ")

                    for point in obj.get('polygon', []):
                        x, y = normalize_coordinates(point[0], point[1], data['imgWidth'], data['imgHeight'])
                        if x is not None and y is not None:
                            f.write(f"{x} {y} ")

                    f.write("\n")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Example usage
def convert_folder_to_yolo(json_folder):
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)
        convert_to_yolo(json_path, json_folder)


#%%
#--------------------------------------------
#%%
# 图片格式转换
def convert_png_to_jpg(path):
    """
    将指定路径下所有的 PNG 图片转换为 JPG 格式
    Args:
        path (str): 待转换图片存放的路径
    """
    # 遍历指定文件夹内所有的 PNG 图片
    try:
        # 遍历指定文件夹内所有的 PNG 图片
        for file_name in os.listdir(path):
            if file_name.endswith('.png'):
                # 读取 PNG 图片
                png_image = Image.open(os.path.join(path, file_name))

                # 将 PNG 图片转换为 JPG 格式
                jpg_image = png_image.convert('RGB')

                # 修改文件后缀名为 JPG
                new_file_name = os.path.splitext(file_name)[0] + '.jpg'

                # 保存转换后的 JPG 图片
                jpg_image.save(os.path.join(path, new_file_name), 'JPEG')

                # 删除原始的 PNG 图片
                os.remove(os.path.join(path, file_name))
        
        st.success("Conversion completed successfully.")
    except FileNotFoundError:
        st.error(f"Error: Directory '{path}' not found.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

#%%


#%%

# 分分类split
def split_datasets(origin_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split a dataset into training, validation, and test sets.

    Args:
    - origin_path (str): Path to the original dataset.
    - train_ratio (float): Ratio of the training set (default is 0.8).
    - val_ratio (float): Ratio of the validation set (default is 0.1).
    - test_ratio (float): Ratio of the test set (default is 0.1).

    Returns:
    - None
    """

    # 设置随机种子
    random.seed(0)

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()
    data_root = origin_path
    origin_flower_path = os.path.join(data_root, "three")
    
    if not os.path.exists(origin_flower_path):
        st.error("Error: path '{}' does not exist.".format(origin_flower_path))
        return

    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # 建立保存训练集、验证集和测试集的文件夹
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    test_root = os.path.join(data_root, "test")

    for path in [train_root, val_root, test_root]:
        if not os.path.exists(path):
            os.makedirs(path)

    for cla in flower_class:
        # 建立每个类别对应的文件夹
        for root in [train_root, val_root, test_root]:
            os.makedirs(os.path.join(root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        train_index = random.sample(images, k=int(num * train_ratio))
        val_index = random.sample(list(set(images) - set(train_index)), k=int(num * val_ratio))
        test_index = list(set(images) - set(train_index) - set(val_index))

        for index, image in enumerate(images):
            if image in train_index:
                new_path = os.path.join(train_root, cla)
            elif image in val_index:
                new_path = os.path.join(val_root, cla)
            else:
                new_path = os.path.join(test_root, cla)

            image_path = os.path.join(cla_path, image)
            try:
                shutil.copy(image_path, new_path)
                st.write("[{}] processing [{}/{}]".format(cla, index + 1, num))  # processing bar
            except Exception as e:
                st.error(f"An error occurred while copying {image}: {e}")
        st.success("Processing done!")


#%%
#改变类别编号
def replace_category(file_path, target_category, new_category):
    try:
        lines = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
     
        new_lines = []
        for line in lines:
            parts = line.split(' ')
            category = int(parts[0])
            if category == target_category:
                parts[0] = str(new_category)
            new_line = ' '.join(parts)
            new_lines.append(new_line)
     
        with open(file_path, 'w') as file:
            file.writelines(new_lines)
        
        st.success("Category replacement completed successfully.")
    except FileNotFoundError as e:
        st.error(f"Error: {e.strerror} {e.filename}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def modify_yolo_txt_files(folder_path, target_category, new_category):
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                replace_category(file_path, target_category, new_category)
        
        st.success("Modification of YOLO text files completed successfully.")
    except FileNotFoundError as e:
        st.error(f"Error: {e.strerror} {e.filename}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
 

# modify_yolo_txt_files(folder_path, target_category, new_category)



#%%33
# 检测split
def split_dataset(image_folder, label_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split a dataset of images and labels into training, validation, and test sets.

    Args:
    - image_folder (str): Path to the folder containing the images.
    - label_folder (str): Path to the folder containing the labels.
    - output_folder (str): Path to the output folder.
    - train_ratio (float): Ratio of the training set (default is 0.8).
    - val_ratio (float): Ratio of the validation set (default is 0.1).
    - test_ratio (float): Ratio of the test set (default is 0.1).

    Returns:
    - None
    """
    try:
        # 创建存放数据集的文件夹
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'val', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'test', 'labels'), exist_ok=True)

        # 获取images文件夹下的所有图片文件
        image_files = [file for file in os.listdir(image_folder) if file.endswith('.jpg')]

        random.shuffle(image_files)

        # 划分数据集
        total_samples = len(image_files)
        train_samples = int(total_samples * train_ratio)
        val_samples = int(total_samples * val_ratio)

        train_images = image_files[:train_samples]
        val_images = image_files[train_samples:train_samples + val_samples]
        test_images = image_files[train_samples + val_samples:]

        # 将图片文件和对应的txt文件复制到相应的文件夹
        def copy_images_and_labels(image_list, set_type):
            for img_file in image_list:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                shutil.copy(os.path.join(image_folder, img_file), os.path.join(output_folder, set_type, 'images', img_file))
                shutil.copy(os.path.join(label_folder, label_file), os.path.join(output_folder, set_type, 'labels', label_file))

        copy_images_and_labels(train_images, 'train')
        copy_images_and_labels(val_images, 'val')
        copy_images_and_labels(test_images, 'test')
        
        st.success("Dataset split completed successfully.")
    except FileNotFoundError as e:
        st.error(f"Error: {e.strerror} {e.filename}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

#%%
def convert_xml_to_yolo(xml_path, image_width, image_height):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        yolo_labels = []

        for object_elem in root.findall('object'):
            name = object_elem.find('name').text
            if name not in classes:
                classes.append(name)
            class_index = classes.index(name)
            xmin = int(object_elem.find('bndbox/xmin').text)
            ymin = int(object_elem.find('bndbox/ymin').text)
            xmax = int(object_elem.find('bndbox/xmax').text)
            ymax = int(object_elem.find('bndbox/ymax').text)

            # Convert bounding box to YOLO format (normalized x,y,w,h)
            x = (xmin + xmax) / 2 / image_width
            y = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_label = f"{class_index} {x} {y} {width} {height}"
            yolo_labels.append(yolo_label)

        return yolo_labels
    except FileNotFoundError:
        st.error(f"Error: File {xml_path} not found.")
        return []
    except Exception as e:
        st.error(f"Error processing file {xml_path}: {str(e)}")
        return []

def convert_dataset_to_yolo(xml_dir, image_dir, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)

        for xml_file in os.listdir(xml_dir):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(xml_dir, xml_file)
                image_name = os.path.splitext(xml_file)[0] + '.jpg'
                image_path = os.path.join(image_dir, image_name)

                image_width, image_height = get_image_size(image_path)
                yolo_labels = convert_xml_to_yolo(xml_path, image_width, image_height)

                yolo_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + '.txt')
                with open(yolo_file, 'w') as f:
                    f.write('\n'.join(yolo_labels))

        st.success("Conversion completed successfully.")
    except FileNotFoundError as e:
        st.error(f"Error: {e.strerror} {e.filename}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def get_image_size(image_path):
    try:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        return width, height
    except FileNotFoundError:
        st.error(f"Error: File {image_path} not found.")
        return (0, 0)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return (0, 0)

# convert_dataset_to_yolo(xml_dir, image_dir, output_dir)
# Usage
#%%5


def remove_extra_files(jpg_folder_path, xml_folder_path):
    """
    Remove extra XML files in the XML folder which do not have corresponding JPG files in the JPG folder.

    Args:
    - jpg_folder_path (str): Path to the JPG folder.
    - xml_folder_path (str): Path to the XML folder.

    Returns:
    - None
    """
    try:
        # 获取jpg文件夹中的文件列表
        jpg_files = [os.path.splitext(file)[0] for file in os.listdir(jpg_folder_path) if file.lower().endswith('.jpg')]
        
        # 获取xml文件夹中的文件列表
        xml_files = [os.path.splitext(file)[0] for file in os.listdir(xml_folder_path) if file.lower().endswith('.xml')]
        
        # 找出在jpg文件夹中存在但在xml文件夹中不存在的文件
        files_to_remove = [os.path.join(xml_folder_path, file + '.xml') for file in xml_files if file not in jpg_files]
        
        # 删除在xml文件夹中多余的文件
        for file_path in files_to_remove:
            os.remove(file_path)
            st.write(f'File "{os.path.basename(file_path)}" deleted from xml folder')
        
        st.success("Extra XML files removed successfully.")
    except FileNotFoundError as e:
        st.error(f"Error: {e.strerror} {e.filename}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


#%%
# 四点voc格式转yolo格式

classes = []
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]

    x = (box[0] + box[2] + box[4] + box[6]) / 4.0
    y = (box[1] + box[3] + box[5] + box[7]) / 4.0

    w = max(box[0], box[2], box[4], box[6]) - min(box[0], box[2], box[4], box[6])
    h = max(box[1], box[3], box[5], box[7]) - min(box[1], box[3], box[5], box[7])

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return x, y, w, h

def convert_annotation(xmlpath, xmlname, imgpath, txtpath, postfix):
    try:
        with open(xmlpath, "r", encoding='utf-8') as in_file:
            txtname = xmlname[:-4] + '.txt'
            txtfile = os.path.join(txtpath, txtname)
            tree = ET.parse(in_file)
            root = tree.getroot()
            img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            res = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    classes.append(cls)
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                x0 = float(xmlbox.find('x0').text)
                y0 = float(xmlbox.find('y0').text)
                x1 = float(xmlbox.find('x1').text)
                y1 = float(xmlbox.find('y1').text)
                x2 = float(xmlbox.find('x2').text)
                y2 = float(xmlbox.find('y2').text)
                x3 = float(xmlbox.find('x3').text)
                y3 = float(xmlbox.find('y3').text)

                b = (x0, y0, x1, y1, x2, y2, x3, y3)

                bb = convert((w, h), b)
                res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
            if len(res) != 0:
                with open(txtfile, 'w+') as f:
                    f.write('\n'.join(res))
    except FileNotFoundError:
        st.error(f"Error: File {xmlpath} not found.")
    except Exception as e:
        st.error(f"Error processing file {xmlpath}: {str(e)}")

def process_files(postfix, imgpath, xmlpath, txtpath):
    if not os.path.exists(txtpath):
        os.makedirs(txtpath, exist_ok=True)
    
    xml_files = [f for f in os.listdir(xmlpath) if f.lower().endswith('.xml')]
    
    for xml_file in xml_files:
        xml_path = os.path.join(xmlpath, xml_file)
        convert_annotation(xml_path, xml_file, imgpath, txtpath, postfix)
    
    st.write(f"Dataset Classes: {classes}")
    
# process_files(postfix, imgpath, xmlpath, txtpath)
#%%
# 抽视频帧做训练集
def extract_frames(video, output_folder, frame_interval):
    
    try:
        # 打开视频文件
        tfile = tempfile.NamedTemporaryFile()
        tfile.write(video.read())
        vid_cap = cv2.VideoCapture(tfile.name)
        frame_count = 0
        success = True
        
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        success = True
        while success:
            success, image = vid_cap.read()
            if success:
                if frame_count % frame_interval == 0:
                    save_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(save_path, image)
                frame_count += 1
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {e}")