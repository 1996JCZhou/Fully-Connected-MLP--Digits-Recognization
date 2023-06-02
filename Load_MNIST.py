import struct
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

"""提取图片和标签信息"""
# 在当前目录/路径/文件夹下, 新建了一个MNIST文件夹, 用于存放各个下载的数据集
# 从当前文件夹/路径(Path())内的MNIST文件夹内取得各个数据集的路径/导入数据集
dataset_path = Path('./MNIST')
train_img_path = dataset_path/'train-images.idx3-ubyte'
train_lab_path = dataset_path/'train-labels.idx1-ubyte'
test_img_path = dataset_path/'t10k-images.idx3-ubyte'
test_lab_path = dataset_path/'t10k-labels.idx1-ubyte'


# 定义解析idx3文件(image文件)的函数
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3图片文件路径
    :return: 数据集
    """
    # open(name, mode)函数: name - 一个包含了要访问的文件名称的字符串值; 
    # mode: 'r' - 以只读方式打开文件; 'rb': 以二进制只读方式打开文件, 读取非人工书写的数据, 如图片;
    # 以二进制只读方式('rb')打开文件, 并将读取的内容放在缓冲区(buffe: bin_data)内, 
    # read()表示全部读取。
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # struct.unpack_from(fmt=,buffer=,offfset=)函数: 从一个二进制文件中读取的内容进行解析操作。
    # 将缓冲区buffer中的二进制文件按照指定的格式fmt='somenformat', 
    # 从偏移量offset=numb的位置开始进行读取, 返回一个元组tuple来存储解析得到的文件。
    # 已知文件头信息包含4个重要信息(数据类型为integer), 因此分别将信息转化为integer。
    # 'i'代表整型integer, 一个'i'占4个字节, 比如： '>iiii'总共占16个字节。
    fmt_header = '>iiii'
    offset = 0
    # 解析文件头信息(这里并不包括图片)，依次为魔数、图片数量、每张图片高度、每张图片宽度
    # 中性词'魔数': 某些具有特定格式的文件，喜欢在文件开头写几个特殊的字符以表明自己的身份，以便验明正身。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('Loading images...')
    print('Magic Nummer: {}, Gesamtzahl der Bilder: {}, Größe der Bilder: {}*{}'.format(magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    # 定义一张照片的尺寸大小
    image_size = num_rows * num_cols
    # 创建一个3维数组, 数组的元素不为空，为随机产生的数据, 用于存储num_images张num_rows * num_cols的图片。
    images = np.empty((num_images, num_rows, num_cols))
    
    # '指针': 得到文件头信息后, 开始解析图片。
    # struct.calcsize()函数: 返回格式字符串fmt描述的结构的字节大小。
    # 从偏移量offset=struct.calcsize(fmt_header)的位置开始进行读取,
    # 即从文件头信息后的第一个pixel开始解析。
    offset += struct.calcsize(fmt_header)
    
    # 内容解析格式: '>'表示的是大端法则('从大往小/从左到右'), 'content_numb'表示的是多少个字节byte,
    # 'B'表示的是一个字节byte的integer。
    # 每个pixel取值从0到255(对应2的8次方), 因此每个pixel占1个字节(8个bit)/1 byte, 8 bits。
    # 一张图片的大小是image_size=28*28, 总共有784个pixel, 总共占784个byte。
    # 联系struct.unpack_from()函数, 一次解析出784个bytes, 即一张图片。
    fmt_image = f'>{image_size}B'
    # 或者: '>' + str(image_size) + 'B' (image_size作为变量)
    # 注: 这里千万不可以写成: '>{image_size}B', 因为这样image_size作为一个变量不会被识别。
    
    for i in range(num_images):
        # '进度提示'
        if (i + 1) % 10000 == 0:
            print(f'{i+1} Bilder geladen')
        # 先将一个元组tuple转化成一个一维数组1D np.array, 然后再将这个一维数组转化成一个28*28的二维数组来表示一张图片。
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # '指针'往后推28*28(784)个位置, 以便于一张接着一张图片解析。
        offset += struct.calcsize(fmt_image)
    print(f'Insgesamt {images.shape[0]} Bilder, Größe der Bilder: {images.shape[1:]}, Größe der Daten von Bildern: {images.shape}。')
    print('\n')
    # 返回一个三维数组, 格式为: (一共多少张图片, 每张图片的行数, 每张图片的列数)。
    return images


# 定义解析idx1文件(label文件)的函数
def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # open(name, mode)函数: name - 一个包含了要访问的文件名称的字符串值; 
    # mode: 'r' - 以只读方式打开文件; 'rb': 以二进制只读方式打开文件, 读取非人工书写的数据, 如图片;
    # 以二进制只读方式('rb')打开文件, 并将读取的内容放在缓冲区(buffe: bin_data)内, 
    # read()表示全部读取。
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    # struct.unpack_from(fmt=,buffer=,offfset=)函数: 从一个二进制文件中读取的内容进行解析操作。
    # 将缓冲区buffer中的二进制文件按照指定的格式fmt='somenformat', 
    # 从偏移量offset=numb的位置开始进行读取, 返回一个元组tuple来存储解析得到的文件。
    # 已知文件头信息包含2个重要信息(数据类型为integer), 因此分别将信息转化为integer。
    # 'i'代表整型integer, 一个'i'占4个字节, 比如： '>ii'总共占8个字节。
    offset = 0
    fmt_header = '>ii'
    # 解析文件头信息(这里并不包括label)，依次为魔数、标签数量
    magic_number, num_labels = struct.unpack_from(fmt_header, bin_data, offset)
    print('loading labels...')
    print('Magic Numemr:{}, Gesamtzahl der Labels: {}个'.format(magic_number, num_labels))

    # 解析数据集
    # 创建一个一维整型空数组, 用于存储num_labels个数据类型为整型的标签。
    # 注: numpy.zeros(shape, dtype=float)函数默认创建数组后的元素数据类型是float,
    # 如果不额外定义数组的元素类型是整型: dtype = int, 那么函数会默认为float浮点型。
    # 参考后续one_hot_label()函数, 因为只有整型数据才能作为位置参数index！
    labels = np.zeros(num_labels, dtype=int)
    
    # '指针': 得到文件头信息后, 开始解析图片。
    # struct.calcsize()函数: 返回格式字符串fmt描述的结构的字节大小。
    # 从偏移量offset=struct.calcsize(fmt_header)的位置开始进行读取,
    # 即从文件头信息后的第一个pixel开始解析。
    offset += struct.calcsize(fmt_header)
    
    # 每个标签label占了一个byte的大小。
    fmt_label = '>1B'
    
    for i in range(num_labels):
        # '进度提示'
        if (i + 1) % 10000 == 0:
            print(f'{i+1} Labels geladen')
        # 创建具有一个元素的元组有点麻烦, 我们需要在表示它是元组的元素后使用尾随逗号。
        # 即使仅有一个数据也会被解包成元组。
        # 只有一个元素的元组: 比如: (7, ), ('Hello!')
        labels[i] = struct.unpack_from(fmt_label, bin_data, offset)[0]
        # '指针'往后推一个位置, 一个接着一个标签解析。
        offset += struct.calcsize(fmt_label)
    print(f'Insgesamt {labels.size} Labels, Größe der Daten von Labels: {labels.shape}。')
    print('\n')
    # 返回一个一维整型数组。
    return labels


# 默认输入参数是train_img_path, 调用的时候不用再输入参数了,
# 直接: load_train_images()
def load_train_images(idx_ubyte_file = train_img_path):
    return decode_idx3_ubyte(idx_ubyte_file)


# 默认输入参数是train_lab_path, 调用的时候不用再输入参数了,
# 直接: load_train_labels()
def load_train_labels(idx_ubyte_file = train_lab_path):
    return decode_idx1_ubyte(idx_ubyte_file)


# 默认输入参数是test_img_path, 调用的时候不用再输入参数了,
# 直接: load_test_images()
def load_test_images(idx_ubyte_file = test_img_path):
    return decode_idx3_ubyte(idx_ubyte_file)


# 默认输入参数是test_lab_path, 调用的时候不用再输入参数了,
# 直接: load_test_labels()
def load_test_labels(idx_ubyte_file = test_lab_path):
    return decode_idx1_ubyte(idx_ubyte_file)


def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()
    #  查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print( train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
        print( 'done')


if __name__ == '__main__':
    run()
    