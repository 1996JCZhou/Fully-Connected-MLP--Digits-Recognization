import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QSize


class PaintBoard(QWidget):
    def __init__(self, Size=QSize(320, 240), Fill=QColor(0, 0, 0, 255)):
        super().__init__()
        """窗口初始化"""
        # 定义窗口的操作界面的尺寸: QSize(int width, int height)
        self.__size = Size
        # 定义窗口的操作界面的默认填充颜色: QColor(int r, int g, int b, int 透明度a=255)
        self.__fill = Fill

        """画笔初始化"""
        # 初始化一个画笔
        self.__painter = QPainter()
        # 默认画笔粗细
        self.__thickness = 18
        # 默认画笔颜色
        self.__penColor = QColor(255, 255, 255, 255)

        """起点和终点初始化"""
        # 构造横纵坐标均为0的QPoint对象: 一个坐标点, 坐标值为int。
        # constructs a null point, i.e. with coordinates (0, 0)
        self.__begin_point = QPoint()
        self.__end_point = QPoint()

        """画板初始化"""
        # 初始化画板为一个像素图, 大小和操作界面的大小一致
        self.__paintboard = QPixmap(self.__size)
        # 初始化画板的操作界面的默认填充颜色: QColor(int r, int g, int b, int 透明度a=255)
        self.__paintboard.fill(self.__fill)
        # 固定住画板的尺寸大小, 不允许放大或缩小画板的大小。
        self.setFixedSize(self.__size)

    # 清空/初始化画板
    def Clear(self):
        self.__paintboard.fill(self.__fill)
        self.update()

    # 定义画板的填充
    def setBoardFill(self, fill):
        self.__fill = fill
        self.__paintboard.fill(fill)
        self.update()

    # 定义画笔的颜色
    def setPenColor(self, color):
        self.__penColor = color

    # 定义画笔的粗细(默认画笔的粗细为10)
    def setPenThickness(self, thickness=10):
        self.__thickness = thickness

    # 将画板上的像素图转换成QImage类型图片
    def getContentAsQImage(self):
        image = self.__paintboard.toImage()
        return image

    # 激活绘画这个行为
    # "constructs a painter that begins painting immediately"
    def paintEvent(self, paintEvent):
        self.__painter.begin(self)
        self.__painter.drawPixmap(0, 0, self.__paintboard)
        self.__painter.end()

    # 接收'按下鼠标左键后释放'这个行为
    # 按下鼠标左键的一瞬间分别刷新画笔的起点和终点位置为当前鼠标的位置, 摆脱了初始化所设定的画笔起点与重点的位置。
    # mousePressEvent()方法比mouseMoveEvent()方法优先被运行, 因此画笔的起点被优先重新设置, 不会造成误操作。
    def mousePressEvent(self, mouseEvent):
        # 按下鼠标左键的瞬间
        if mouseEvent.button() == Qt.LeftButton:
            # pos()函数: returns the position of the mouse cursor
            # (在按下做鼠标左键的一瞬间)使用鼠标当前位置定义画笔的起点
            self.__begin_point = mouseEvent.pos()
            print(f'begin point: {self.__begin_point}')
            # 画笔的起点=画笔的终点=鼠标当前的位置
            self.__end_point = self.__begin_point
            print(f'end point: {self.__end_point}')
            self.update()

    # 接收移动鼠标这个行为
    # If mouse tracking is switched off, mouse move events only occur
    # if a mouse button is pressed while the mouse is being moved.
    def mouseMoveEvent(self, mouseEvent):
        # 按下鼠标左键的瞬间
        if mouseEvent.buttons() == Qt.LeftButton:
            # 在按下鼠标左键的一瞬间鼠标移动了'一小步', 移动后的位置被设定为画笔的终点, 配合后续作画。
            self.__end_point = mouseEvent.pos()

            # 开始在画板上作图
            self.__painter.begin(self.__paintboard)
            # setPen(style): sets the painter’s pen to have the given style
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))
            # drawLine(start, end)函数: 画一条从起点指向终点的直线。
            self.__painter.drawLine(self.__begin_point, self.__end_point)
            self.__painter.end()

            # 将画笔的终点定义为画笔的起点, 为下次作图做准备。
            self.__begin_point = self.__end_point
            self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = PaintBoard()
    demo.show()
    sys.exit(app.exec_())
