"""
1、先是用yolo 获得目标位置及其标签
2、把图片按一定标准剪切，目前暂定为四等分剪切
3、判断剪切部分有无目标
4、将判断结果（目标有无、若有即为种类，其中-1表示没有目标，其它的数字表示目标的种类）储存在label中
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob


# 将两个矩形是否相交转化为四个点abcd是否落在矩形内
def is_points_in_rectangle(points, rectangle, img_type, i):
    # points为四个点的坐标，格式为[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # rectangle为矩形E的左上顶点和右下顶点的坐标，格式为[(left, top), (right, bottom)]
    count = 0
    for point in points:
        x, y = point
        left, top = rectangle[0]
        right, bottom = rectangle[1]
        if x >= left and x <= right and y >= top and y <= bottom:
            count += 1

    if count > 0:
        print(f"子图片{i}包含目标，顶点个数为、类别为:", count, img_type)
        return  img_type
    else:
        print(f"子图片{i}没有目标")
        return -1  #-1代表没有目标




class Trim:
    def __init__(self, image_path, labelfolder_path, image_name, save_folder):
        self.image_path = image_path
        self.labelfolder_path = labelfolder_path
        self.image_name = image_name
        self.save_folder = save_folder


    def center_label(self):  # 读取label的中心点数据
        img = cv2.imread(self.image_path)  # 图片尺寸统一为256列 x 155行
        if img is None:
            print("Failed to load image")
        else:
            print("Image loaded successfully")
        # 获取图片的宽度和高度
        height, width = img.shape[:2]
        # width, height = img.shape[1], img.shape[0]
        # 读取指定文件夹里面的文件
        txt_name = self.image_name + '.txt'
        folder_path = self.labelfolder_path
        txt_name = os.path.join(folder_path, txt_name)
        # 开始读取txt文件里面的内容
        with open(txt_name, 'r') as f:
            for line in f:
                data = line.strip().split(' ')
                image_type = data[0]
                a, b, c, d = int((float(data[1]) - 0.5 * float(data[3])) * width), int(
                    (float(data[1]) + 0.5 * float(data[3])) * width), int(
                    (float(data[2]) - 0.5 * float(data[4])) * height), int(
                    (float(data[2]) + 0.5 * float(data[4])) * height)
                # 左上顶点（a，c）a:左上列号，c，左上行号  右下顶点（b，d）b:右下列号，d：右下行号
        return image_type, a, c, b, d

    def trimming_new(self):
        img, height, width = self.read_img()  # print(height,width)#155 256
        image_type, x1, y1, x2, y2 = self.center_label()  # 输入图片类别\左上顶点和右下顶点坐标
        #######################################################
        # 在图片上绘制矩形框,如果不要矩形就把这个引起来
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.axis('off')
        #plt.show()

        bigger = 5
        #在不超过图片尺寸的前提下扩大区域
        y1 = y1-bigger if y1-bigger >= 0 else y1
        x1 = x1 - bigger if x1 - bigger >= 0 else x1
        y2 = y2 + bigger if y2 + bigger <= height else y2
        x2 = x2 + bigger if x2 + bigger <= width else x2

        img_range = img[y1:y2, x1:x2]
        self.save_img_new(img_range, image_type)#保存图片


    def check_folder(self,save_folder):
        # 检查文件夹是否已经存在
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f"创建文件夹: {save_folder}")  # 创建子文件夹
        else:
            print(f"该文件夹已存在: {save_folder}")

    def save_img_new(self,img,image_type):
        # 子文件夹路径
        subfolder_path = os.path.join(self.save_folder, 'jubu')
        save_imagefolder = os.path.join(subfolder_path, 'image')
        save_labelfolder = os.path.join(subfolder_path, 'label')
        self.check_folder(save_imagefolder)
        self.check_folder(save_labelfolder)
        """
        #保存剪裁好的图片
        imgs_path = os.path.join(save_imagefolder, self.image_name + '.jpg')
        cv2.imwrite(imgs_path, img)
        print("finish---save image.")
        
        # 保存剪裁好的图片的label
        txt_path = os.path.join(save_labelfolder, self.image_name + '.txt')
        if not os.path.exists(txt_path):
            # 如果txt文件不存在，则创建该文件
            with open(txt_path, 'w') as file:
                file.write(str(image_type))
            # print(f"创建并写入 '{image_types[n]}' to {txt_path}")
        else:
            # 如果txt文件已存在，则将imgtype写入该文件
            with open(txt_path, 'a') as file:
                file.write(str(image_type))
        print("finish---save label.")
        """
        #按照类别储存图片
        class_path = os.path.join(save_imagefolder, str(image_type))
        self.check_folder(class_path)
        imgs_path = os.path.join(class_path, self.image_name + '.jpg')
        cv2.imwrite(imgs_path, img)
        print("finish---save image.")

        #label也按类别
        class_path2 = os.path.join(save_labelfolder, str(image_type))
        self.check_folder(class_path2)
        txt_path = os.path.join(class_path2, self.image_name + '.txt')
        if not os.path.exists(txt_path):
            # 如果txt文件不存在，则创建该文件
            with open(txt_path, 'w') as file:
                file.write(str(image_type))
            # print(f"创建并写入 '{image_types[n]}' to {txt_path}")
        else:
            # 如果txt文件已存在，则将imgtype写入该文件
            with open(txt_path, 'a') as file:
                file.write(str(image_type))
        print("finish---save label.")


    def read_img(self):
        # 使用OpenCV读取图片
        img = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), 1)
        if img is None:
            print("Failed to load image")
        else:
            print("Image loaded successfully")
        # 获取图片的宽度和高度
        height, width = img.shape[:2]
        return img, height, width



#image_path = "train/images/1_0.jpg"
#image_name = "1_0"
imagesfolder="airplane/images/train2014"
labelfolder_path = "airplane/labels/train2014"
save_folder = "sava"#"result"#保存数据的文件夹
def read_imagesfolder(folder_path):
    # 获取文件夹中所有图片的路径
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))

    for image_path in image_paths:
        # 获取图片的名称（去除尾缀）
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 输出图片的路径和名称
        #print("Image Path:", image_path)
        #print("Image Name:", image_name)
        t = Trim(image_path, labelfolder_path, image_name, save_folder)
        t.center_label()
        #t.trimming()
        t.trimming_new()
read_imagesfolder(imagesfolder)