import sys
import numpy                as np
from PIL                    import Image, ImageQt
from Layout                 import Ui_Main_Window
from PaintBoard             import PaintBoard
from PyQt5.QtWidgets        import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtGui            import QPixmap, QColor
from PyQt5.QtCore           import QSize

"""MNN"""
import math
from Net_algo               import Net

net = Net(Shape=[784, 100, 10], Distribution=[
                                              {},
                                              {'w': [-math.sqrt(6/(784+100)), math.sqrt(6/(784+100))], 'b': [0, 0]},
                                              {'w': [-math.sqrt(6/(100+10)), math.sqrt(6/(100+10))], 'b': [0, 0]},
                                             ], Batch_size=50)
"""Initialisierung"""
net.init_parameters()
net.print_parameter()

"""Datenvorbereitung"""
temp_train_images, train_labels = net.load_training_data()
train_images = net.gaussian(temp_train_images)
temp_test_images, test_labels = net.load_test_data()
test_images = net.gaussian(temp_test_images)

"""Training"""
Parameters = net.training(train_images, train_labels, test_images, test_labels)

"""GUI"""
MODE_MNIST = 1
MODE_WRITE = 2


class Main_Window(QMainWindow, Ui_Main_Window):

    def __init__(self):
        super(Main_Window, self).__init__()

        # 初始化参数
        self.mode = MODE_MNIST
        self.result = [0, 0]

        # 初始化UI
        self.setupUi(self)
        self.center()

        # 添加并初始化画板
        self.paintBoard = PaintBoard(Size=QSize(224, 224),
                                     Fill=QColor(0, 0, 0, 0))
        self.dArea_Layout.addWidget(self.paintBoard)

        self.clearDataArea()

    # 窗口居中
    def center(self):
        # 按照frameGeometry()的大小创建窗口。
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 将窗口显示/移动到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

    # 窗口关闭事件: 当关闭窗口时, 特定的事件会出现。
    # Close events are sent to widgets that the user wants to close,
    # usually by choosing “Close” from the window menu,
    # or by clicking the X title bar button.
    def closeEvent(self, event):
        event.accept()

    # 清除数据待输入区
    def clearDataArea(self):
        # 清空/初始化画板
        self.paintBoard.Clear()
        # clear widget: Image_Show
        self.Image_Show.clear()
        # clear widget: Predict_Result
        self.Predict_Result.clear()
        # clear widget Softmax_Result
        self.Softmax_Result.clear()
        self.result = [0, 0]

    # 数据清除
    def pbtClear_Callback(self):
        self.clearDataArea()

    # 从(未经过标准化处理的)测试集中随机抽取一张图片显示(用来检测)
    def pbtGetMnist_Callback(self):
        # 先"全场"清零"。
        self.clearDataArea()

        # 再随机抽取一张测试集中的图片, 经过操作后直接呈现在标签Image_Show上。
        # 随机抽取一张测试集中的图片imgimg
        random_num = np.random.randint(0, temp_test_images.shape[0])
        array_img = temp_test_images[[random_num]]
        # numpy array with Shape: (28, 28)
        array_img = array_img.reshape(28, 28)
        # 实现从numpy数组array到PIL的图像数据类型image的转换
        temp_img = Image.fromarray(np.uint8(array_img))
        # 将iamge放大, 尺寸与Image_Show标签label一致, 用于投放
        temp_img = temp_img.resize((221, 221))
        # PIL中的图片数据类型Image将转换成PyQt5中的图片数据类型QImage
        qimage = ImageQt.ImageQt(temp_img)
        # 将QImage类型图像转换为像素图类型图像
        pix_img = QPixmap.fromImage(qimage)
        # 将像素图类型图像显示在label: Image_Show上
        self.Image_Show.setPixmap(pix_img)

    # 模式下拉列表回调
    def cbBox_Mode_Callback(self, text):
        # 当下拉菜单上的选项/文字描述为: ...
        if text == '1: random pick one picture from MNIST test set':
            self.mode = MODE_MNIST
            self.clearDataArea()
            # 激活PushButton按钮MNIST_Random_Draw
            self.MNIST_Random_Draw.setEnabled(True)
            # 将画板的背景填充色: 白 白 白 透明
            self.paintBoard.setBoardFill(QColor(0, 0, 0, 0))
            # 将画板的画笔填充色: 白 白 白 透明
            self.paintBoard.setPenColor(QColor(0, 0, 0, 0))

        elif text == '2: draw one digit in the Input Data Space':
            self.mode = MODE_WRITE
            self.clearDataArea()
            # 关闭PushButton按钮MNIST_Random_Draw
            self.MNIST_Random_Draw.setEnabled(False)
            # 将画板的背景填充色: 黑 黑 黑 完全不透明
            self.paintBoard.setBoardFill(QColor(0, 0, 0, 255))
            # 将画板的画笔填充色: 白 白 白 blur=0(不完全不透明)
            self.paintBoard.setPenColor(QColor(255, 255, 255, 150))

    # 识别图片
    def pbtPredict_Callback(self):
        # 初始化空列表
        pix__img = []

        if self.mode == MODE_MNIST:
            # 读取在label: Image_Show上的像素图, 并保存在pix__img列表内
            pix__img = self.Image_Show.pixmap()
            # 若在没有随机挑选图片的情况下就开始识别, 那么label内无图像。
            # 若label内无图像, 则返回None。
            if pix__img == None:   # 无图像则用纯黑代替
                # 通过Image.fromarray(np.uint8(np.zeros([224, 224])))创建一个实例: QImage类型图像
                pix__img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            # 将画板上的像素图转换成QImage类型图片
            else: pix__img = pix__img.toImage()
        elif self.mode == MODE_WRITE:
            # 将画板上的像素图转换成QImage类型图片
            pix__img = self.paintBoard.getContentAsQImage()

        # 将PyQt5中的图片数据类型QImage转换成PIL中的图片数据类型Image
        pil_img = ImageQt.fromqimage(pix__img)
        # Image.ANTIALIAS: smooth filter, high-quality resampling filtering for all input pixels
        # pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
        pil_img = pil_img.resize((28, 28))
        # 将PIL中的图片数据类型Image转换成numpy数组array, 尺寸大小不变
        temp_array_img = np.array(pil_img.convert('L'))
        # 将numpy数组array改变尺寸成'一行'的二维数组
        array_img = temp_array_img.reshape(1, 28*28)
        # 将numpy数组array进行标准化处理
        test_img = net.gaussian(array_img)

        text_result = net.FP(test_img, Parameters)
        self.result[0] = np.argmax(text_result)
        self.result[1] = text_result[0, self.result[0]]

        self.Predict_Result.setText("%d" % (self.result[0]))
        self.Softmax_Result.setText("%.8f" % (self.result[1]))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Window = Main_Window()
    Window.show()
    sys.exit(app.exec_())
