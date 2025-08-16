from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QGuiApplication, QCursor
import os
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import *
import cv2
import shutil
from collections import deque



class MyLabel(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    qlabel_height = 0
    height = 0
    width = 0
    txt_x0 = 0
    txt_y0 = 0
    txt_x1 = 0
    txt_y1 = 0


    def __init__(self,parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0

    def wheelEvent(self, event):#放大输出的行列号是不对的
        # Get the current position of the mouse
        mouse_pos = event.position().toPoint()

        # Scale the image
        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1

        # Set the new scale factor
        self.setScaledContents(True)
        self.setPixmap(self.pixmap().scaled(self.scale_factor * self.pixmap().size()))

        # Adjust the position of the mouse to keep it centered
        new_mouse_pos = self.mapFromGlobal(QCursor.pos())
        delta = new_mouse_pos - mouse_pos
        self.move(self.pos() - delta)



    # 鼠标点击事件
    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.flag = False

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    # 绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        # print(self.x0, self.y0, self.x1, self.y1)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawRect(rect)
        # print("x0, y0 ,x1, y2:", self.x0, self.y0, self.x1, self.y1)
        pqscreen = QGuiApplication.primaryScreen()
        pixmap2 = pqscreen.grabWindow(self.winId(), self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        x2, y2, x3, y3 = 0, int(self.qlabel_height / 2) - int(self.height / 2), int(self.width), int(self.height)
        # print('x2,y2,x3,y3:', x2, y2, x3, y3)
        pixmap2 = pqscreen.grabWindow(self.winId(), 0, y2, x3, y3)
        # pixmap2 = pqscreen.grabWindow(self.winId(), 0, int(1150 / 2) - int(155 / 2), 256, 155)
        pixmap2.save('555.jpg')
        self.txt_x0, self.txt_y0 = self.x0, self.y0 - y2
        self.txt_x1, self.txt_y1 = self.txt_x0 + abs(self.x1 - self.x0), self.txt_y0 + abs(self.y1 - self.y0)
        # print("txt_label的左上列行索引和右下列行索引为：",self.txt_x0, self.txt_y0,self.txt_x1, self.txt_y1 )
        # 打开文件，如果文件不存在则创建
        with open('666.txt', 'w') as f:
            # 将四个变量的值写入文件中
            f.write('{} {} {} {}'.format(self.txt_x0, self.txt_y0, self.txt_x1, self.txt_y1))
        # print('aaaaaaaaaaaaaaaaaaa')


class Auto:
    def __init__(self, path, labelfolder_path, name, count, picture_type):
        self.path = path  # path为图片路径
        self.labelfolder_path = labelfolder_path
        self.name = name  ## name为图片名称，无JPG
        self.count = count  # 记录读取了多少张图片
        self.picture_type = picture_type  # 记录目标的类别

    def open_g(self):  # 读取图片
        print('读取的图片个数：', self.count)
        img = cv2.imread(self.path)  # 图片尺寸统一为256列 x 155行
        # cv2.namedWindow(self.path, cv2.WINDOW_FREERATIO)
        # cv2.moveWindow(self.path, 500, 250)
        # cv2.imshow(self.path, img)
        # cv2.waitKey(0)
        matrix_g = img[:, :, 1]  # 单通道读取
        self.matrix_g = matrix_g
        self.img = img
        return self.matrix_g, self.img

    def center_label(self):  # 读取label的中心点数据
        img = cv2.imread(self.path)  # 图片尺寸统一为256列 x 155行
        width, height = img.shape[1], img.shape[0]
        # 读取指定文件夹里面的文件
        txt_name = self.name + '.txt'
        ###########
        folder_path = self.labelfolder_path
        txt_name = os.path.join(folder_path, txt_name)
        # 开始读取txt文件里面的内容
        with open(txt_name, 'r') as f:
            for line in f:
                data = line.strip().split(' ')
                type, start_x, start_y = data[0], int(float(data[1]) * width), int(float(data[2]) * height)
                a, b, c, d = int((float(data[1]) - 0.5 * float(data[3])) * width), int(
                    (float(data[1]) + 0.5 * float(data[3])) * width), int(
                    (float(data[2]) - 0.5 * float(data[4])) * height), int(
                    (float(data[2]) + 0.5 * float(data[4])) * height)
                # 左上顶点（a，c）a:左上列号，c，左上行号  右下顶点（b，d）b:右下列号，d：右下行号
        # 保存画了中心点的图片
        # img2 = cv2.circle(img, (start_x, start_y), 1, (0, 0, 255), -1)  # 画中心点# 出发点为红色
        # folder_path = "after_center"
        # img_name = os.path.join(folder_path, self.name + '.jpg')
        # cv2.imwrite(img_name, img2)
        return start_x, start_y, a, b, c, d

    def difference(self, x, y, next_x, next_y, matrix):
        if abs((matrix[y][x] - matrix[next_y][next_x])) <= 45:  # 差值在20以内，可接受
            return 0  # 通过
        else:
            return 1

    def rate(self, x, y, next_x, next_y, matrix):
        if matrix[next_y][next_x] / matrix[y][x] >= 0.99:
            return 0  # 通过
        else:
            return 1

    def rad(self, x, y, next_x, next_y, matrix):
        if matrix[next_y][next_x] / matrix[y][x] >= 0.975 and abs((matrix[y][x] - matrix[next_y][next_x])) <= 25:
            return 0  # 通过
        else:
            return 1

    def ratee(self, x, y, next_x, next_y, matrix):
        if matrix[next_y][next_x] / matrix[y][x] >= 0.99 and matrix[next_y][next_x] / matrix[y][x] <= 1.02:
            return 0  # 通过
        else:
            return 1

    def yuzhi(self, x, y, next_x, next_y, matrix):
        if matrix[next_y][next_x] >= 120 and matrix[next_y][next_x] <= 225:
            return 0  # 通过
        else:
            return 1

    def bfs_matrix(self, start_x, start_y, img, matrix):
        """

        :param start_x: 中心点列号
        :param start_y: 中心点行号
        :param img: 图片
        :param matrix:图片的通道值
        :return: 区域的两顶点的坐标（列，行）
        """
        print("起始点为：", start_x, start_y)
        width, height = img.shape[1], img.shape[0]
        print(f"列号：%d,行号：%d", width, height)
        vis = [[False] * width for _ in range(height)]  # width列号
        queue = deque([(start_x, start_y, 0)])
        vis[start_y][start_x] = True
        max_x, min_y, max_y, min_x = start_x, start_y, start_y, start_x  # 前两个分别代表左上角的列号、行号
        # 设置相邻的八个方向的走向
        go = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]]
        while queue:
            x, y, steps = queue.popleft()
            for next in go:
                next_x, next_y = x + next[0], y + next[1]
                if vis[next_y][next_x] is False and self.yuzhi(x, y, next_x, next_y, matrix) == 0:
                    if max_x <width and min_x>0 and max_y<height and min_y>0:
                        vis[next_y][next_x] = True
                        queue.append((next_x, next_y, steps + 1))
                        max_x, min_x = max(next_x, max_x), min(next_x, min_x)  # 右下和左上列号
                        max_y, min_y = max(next_y, max_y), min(next_y, min_y)
        return min_y, min_x, max_y, max_x

    def rectangele_save(self, min_y, min_x, max_y, max_x, img, x, y):  # 画矩形#并保存左上角和右上角坐标
        # 画矩形，矩形为黄色，因为min_y, min_x, max_y, max_x实际是目标边界（目标的一部分），因此向外偏移一个像素画矩形
        # cv2.namedWindow(self.path, cv2.WINDOW_FREERATIO)
        # cv2.moveWindow(self.path, 500, 250)
        width, height = img.shape[1], img.shape[0]
        if (max_x - min_x) * (max_y - min_y) <= width * height / 100 and (max_x - min_x) * (max_y - min_y) > 0:
            img2 = cv2.rectangle(img, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), (0, 255, 255), 1, 4)
            img2 = cv2.circle(img2, (x, y), 1, (0, 0, 255), -1)
            folder_path = "after_rectangle"
            img_name = os.path.join(folder_path, self.name + '.jpg')
            cv2.imwrite(img_name, img2)
            # 将图片名称和两顶点的坐标存入txt中，并将所有txt文件存储到label文件里
            min_y, min_x, max_y, max_x = str(min_y), str(min_x), str(max_y), str(max_x)
            src_file = f"{self.name}.txt"
            dst_folder = 'label'
            with open(src_file, "w") as f:
                f.write(f"{self.picture_type}" + ' ' + min_x + ' ' + min_y + ' ' + max_x + ' ' + max_y + '\n')
            # 如果目标文件夹中已经存在同名txt文件，则删除原文件
            dst_file = os.path.join(dst_folder, os.path.basename(src_file))

            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.move(src_file, dst_folder)
        # 画中心点# 出发点为红色
        # cv2.imshow(self.path, img2)
        # cv2.waitKey(0)
        # 保存画了矩形框的图片
        else:
            folder_path = "origin"
            img_name = os.path.join(folder_path, self.name + '.jpg')
            cv2.imwrite(img_name, img)

    def all(self):
        global maxx, minn, minx
        matrix_g, img = self.open_g()
        # print(matrix_g)
        # start_x, start_y,img = self.center(img)  # start_x为列号
        # start_x, start_y = self.center(img)  # start_x为列号
        start_x, start_y, a, b, c, d = self.center_label()  # start_x为列号
        # graph = self.graph(matrix_g)
        # min_y, min_x, max_y, max_x = self.bfs(start_x, start_y, graph, img)

        min_y, min_x, max_y, max_x = self.bfs_matrix(start_x, start_y, img, matrix_g)

        with open('777.txt', 'w') as f:
            # 将四个变量的值写入文件中
            f.write('{} {} {} {}'.format(min_x, min_y, max_x, max_y))
            print('自动生成的777.txt:', min_x, min_y, max_x, max_y)

        max1 = 0
        min1 = 900
        # print('左上角、右下角:', min_y, min_x, ',', max_y, max_x)
        # print('*' * 30)
        # print('中心点', matrix_g[start_y - 1][start_x - 1])
        for i in range(c, d):  # 原给框太大，进行适当缩小
            for j in range(a, b):
                if matrix_g[i - 1][j - 1] > max1:
                    max1 = matrix_g[i - 1][j - 1]
                    maxx = max(max1, maxx)
                if matrix_g[i - 1][j - 1] < min1:
                    min1 = matrix_g[i - 1][j - 1]
                    minn = min(min1, minn)
                print(matrix_g[i - 1][j - 1], end=' ')
            print()
        minx = min(max1, minx)
        print('max', max1, 'min', min1)  # 每张图片的给定范围内的最大最小值
        print('global max', maxx, 'global min', minn,minx)  # 所有图片的给定范围内的最大最小值

        ''' for i in range(0, 156):
            for j in range(0, 256):
                if matrix_g[i - 1][j - 1] > max1:
                    max1 = matrix_g[i - 1][j - 1]
                    maxx = max(max1, maxx)
                if matrix_g[i - 1][j - 1] < min1:
                    min1 = matrix_g[i - 1][j - 1]
                    minn = min(min1, minn)
                print(matrix_g[i - 1][j - 1], end=' ')
            print()
        print('max', max1, 'min', min1)
        print('global max', maxx, 'global min', minn)'''
        # self.rectangele_save(min_y, min_x, max_y, max_x, img, start_x, start_y)


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        # 当前图片的索引
        self.index = 0

        # 图片文件夹的路径
        self.folder_path = ''

        # 图片文件夹中的所有图片文件名列表
        self.image_files = []

        # 操作图片的label
        self.labelfolder_path = ''

        # 保存图片文件夹的路径
        self.newfolder_path = ''

        # 新图片文件夹中的所有图片文件名列表
        self.newimage_files = []

        # 图片的类型#先默认为是0，等一下和小贾的结合
        self.image_type = 0

        # 0代表只预览了，1代表还重画了
        self.condition = 0

        self.initUI()

    def initUI(self):
        # 标签用于显示当前图片的文件路径
        self.path_label = QLabel(self)
        self.path_label.setFixedHeight(30)
        self.path_label.setGeometry(30, 60, 1000, 60)

        # 标签用于显示新图和新txt的文件夹
        self.savepath_label = QLabel(self)
        self.savepath_label.setFixedHeight(30)
        self.savepath_label.setGeometry(30, 80, 1000, 60)

        # 按钮用于打开文件夹选择框
        self.open_button = QPushButton('图片文件夹', self)
        self.open_button.clicked.connect(self.show_folder_dialog)
        self.open_button.setGeometry(10, 10, 100, 30)

        # 按钮用于展示上一张图片
        self.prev_button = QPushButton('上一张', self)
        self.prev_button.clicked.connect(self.show_prev_image)
        self.prev_button.setGeometry(450, 10, 100, 30)

        # 按钮用于展示下一张图片
        self.next_button = QPushButton('下一张', self)
        self.next_button.clicked.connect(self.show_next_image)
        self.next_button.setGeometry(450, 40, 100, 30)  # 340, 40, 100, 30

        # 按钮用于选定保存新图片的文件夹
        self.savefolder_button = QPushButton('保存文件夹', self)
        self.savefolder_button.clicked.connect(self.save_folder_dialog)
        self.savefolder_button.setGeometry(310, 10, 100, 30)

        # 按钮用于合并自动和预览
        self.hebin_button = QPushButton('先自动画', self)
        self.hebin_button.clicked.connect(self.final_auto)
        self.hebin_button.setGeometry(10, 40, 100, 30)

        # 按钮用于画框
        self.draw_button = QPushButton('手动重画', self)
        self.draw_button.clicked.connect(self.draw)
        self.draw_button.setGeometry(160, 40, 100, 30)
        # self.draw_button.clicked.connect(self.draw_rect)

        # 按钮用于保存当前图片
        self.save_button = QPushButton('保存', self)
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setGeometry(310, 40, 100, 30)  # 230, 40, 100, 30

        # 按钮用于选定操作图片的label文件夹
        self.labelfolder_button = QPushButton('label文件夹', self)
        self.labelfolder_button.clicked.connect(self.label_folder_dialog)
        self.labelfolder_button.setGeometry(160, 10, 100, 30)

        self.type_button = QLabel('类型', self)
        self.type_button.setGeometry(600, 10, 60, 30)

        # 输入对应类型
        self.input_button = QLineEdit(self)
        self.input_button.setGeometry(650, 10, 60, 30)
        self.input_button.setObjectName('input_button')

        self.resize(1150, 1150)

        self.setWindowTitle('标注工具')
        self.lb = MyLabel(self)  # 重定义的label
        self.qlabel = 1150  # 如果图片长宽超过1150，把这个地方改大一点
        self.lb.setGeometry(QRect(100, 50, self.qlabel, self.qlabel))

    def show_folder_dialog(self):
        # 打开文件夹选择框
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')

        if folder_path:
            # 如果选择了文件夹，更新图片文件夹的路径和图片文件名列表
            self.folder_path = folder_path
            self.image_files = sorted(os.listdir(folder_path))

            # 如果图片文件不为空，展示第一张图片
            if self.image_files:
                self.index = 0
                self.show_image()

    def save_folder_dialog(self):
        # 选定保存文件夹
        newfolder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if newfolder_path:
            # 如果选择了文件夹，更新保存图片文件夹的路径和图片文件名列表
            self.newfolder_path = newfolder_path
            self.save_show()

    def label_folder_dialog(self):
        # 选定保存文件夹
        label_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if label_path:
            self.labelfolder_path = label_path

    def draw(self):
        # 手动画框
        self.condition = 1
        image_path = os.path.join(self.folder_path, self.image_files[self.index])
        print('image_path:', image_path)
        img = cv2.imread(image_path)
        height, width, bytesPerComponent = img.shape
        print("height,width", height, width)
        bytesPerLine = 3 * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.lb.qlabel_height = self.qlabel  # 设定最终保存图片的参数
        self.lb.height = height
        self.lb.width = width
        self.lb.setPixmap(pixmap)
        self.lb.setCursor(Qt.CrossCursor)
        self.show()

    def show_prev_image(self):
        # 展示上一张图片
        if self.index > 0:
            self.index -= 1
            self.show_image()
            self.save_show()
            self.lb.x0 = 0
            self.lb.y0 = 0
            self.lb.x1 = 0
            self.lb.y1 = 0
            self.lb.update()
            self.draw()
            self.condition = 0

    def show_next_image(self):
        # 展示下一张图片
        if self.index < len(self.image_files) - 1:
            self.index += 1
            self.show_image()
            self.save_show()
            self.lb.x0 = 0
            self.lb.y0 = 0
            self.lb.x1 = 0
            self.lb.y1 = 0
            self.lb.update()
            self.draw()
            self.condition = 0

    def show_image(self):
        # 更新标签上的图片和文件路径
        image_path = os.path.join(self.folder_path, self.image_files[self.index])
        self.path_label.setText(image_path)

    def save_show(self):
        # 更新保存路径
        newimage_path = os.path.join(self.newfolder_path, self.image_files[self.index])
        self.savepath_label.setText(newimage_path)

    def save_image(self):
        # 把画好的图片放进保存的文件夹里
        newimage_path = os.path.join(self.newfolder_path, self.image_files[self.index])
        self.a = self.input_button.text()
        img = cv2.imread('555.jpg')
        # Save the image to the new directory
        cv2.imwrite(newimage_path, img)

        if self.condition == 0:
            # 源文件路径
            with open('777.txt', 'a+') as f:
                f.seek(0)
                old = f.read()
                f.truncate(0)
                f.write(self.a)
                f.write(' ')
                f.write(old)
            # 写入目标类型
            txt_path = '777.txt'
            # 把jpg命名改为txt：
            newtxt_path = os.path.splitext(newimage_path)[0] + ".txt"
            # 使用shutil.copy()函数将文件从源文件夹复制到目标文件夹
            shutil.copy(txt_path, newtxt_path)
            pass
        else:
            # 源文件路径
            with open('666.txt', 'a+') as f:
                f.seek(0)
                old = f.read()
                f.truncate(0)
                f.write(self.a)
                f.write(' ')
                f.write(old)
            # 写入目标类型
            txt_path = '666.txt'
            # 把jpg命名改为txt：
            newtxt_path = os.path.splitext(newimage_path)[0] + ".txt"
            # 使用shutil.copy()函数将文件从源文件夹复制到目标文件夹
            shutil.copy(txt_path, newtxt_path)
        self.condition = 0

    def final_auto(self):
        self.condition = 0
        # 将选中的文件夹中的所有图片进行自动画框处理
        image_path = os.path.join(self.folder_path, self.image_files[self.index])
        image_name = os.path.splitext(self.image_files[self.index])[0]
        # label_path = os.path.join(self.labelfolder_path, self.image_files[self.index])
        print("image_path,image_name", image_path, image_name, self.index, self.image_type)
        au = Auto(image_path, self.labelfolder_path, image_name, self.index, self.image_type)
        au.all()
        # 把结果生成的矩形的坐标统一储存到某一txt
        newimage_path = os.path.join(self.newfolder_path, self.image_files[self.index])
        txt_path = '777.txt'
        # 把jpg命名改为txt：
        newtxt_path = os.path.splitext(newimage_path)[0] + ".txt"
        # 使用shutil.copy()函数将文件从源文件夹复制到目标文件夹
        shutil.copy(txt_path, newtxt_path)
        # 把处理后的图片展示出来
        with open('777.txt', 'r') as file:
            for line in file:
                data = line.strip().split(' ')
                txt_x0, txt_y0 = int(data[0]), int(data[1])  # 列号 # 行号
                txt_x1, txt_y1 = int(data[2]), int(data[3])

        image_path = os.path.join(self.folder_path, self.image_files[self.index])
        print('image_path:', image_path)
        img = cv2.imread(image_path)
        img = cv2.rectangle(img, (txt_x0, txt_y0), (txt_x1, txt_y1), (0, 255, 255), 1, 4)
        #start_x,start_y=
        #img = cv2.circle(img,)
        print('画在img的坐标：', txt_x0, txt_y0, txt_x1, txt_y1)
        height, width, bytesPerComponent = img.shape
        print("height,width", height, width)
        bytesPerLine = 3 * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.lb.qlabel_height = self.qlabel  # 设定最终保存图片的参数
        self.lb.height = height
        self.lb.width = width
        self.lb.setPixmap(pixmap)
        self.lb.setCursor(Qt.CrossCursor)
        self.show()


if __name__ == '__main__':
    maxx = 0
    minn = 255
    minx = 255
    situation = 0
    app = QApplication([])
    image_viewer = ImageViewer()
    image_viewer.show()
    app.exec_()
