import copy
import math
import numpy    as np
from Load_MNIST import load_train_images, load_train_labels, load_test_images, load_test_labels
from matplotlib import pyplot as plt


class Net:
    def __init__(self, Shape, Distribution, Parameters=None, Batch_size=None, Epoch=None):
        self.dimension = Shape
        self.distribution = Distribution

        """Default"""
        # Initialisierung und Speicherung auf Liste: self.parameters (global)
        self.parameters = []  if Parameters is None else Parameters
        self.batch_size = 100 if Batch_size is None else Batch_size
        self.epoch = 10       if Epoch      is None else Epoch

    """Initialisierung von Gewichtungen und Bias"""
    def init_parameters_b(self, layer):
        # self.distribution - layer - vordefinierter Anfangswertsumfang von Bias
        dist = self.distribution[layer]['b']
        # Bias: 2D Array/Matrix mit Werten in diesem Umfang
        return np.random.random((1, self.dimension[layer]))*(dist[1]-dist[0]) + dist[0]

    def init_parameters_w(self, layer):
        # self.distribution - layer - vordefinierter Anfangswertsumfang von Gewichtungen
        dist = self.distribution[layer]['w']
        # Gewichtungen: 2D Array/Matrix mit Werten in diesem Umfang
        return np.random.random((self.dimension[layer-1], self.dimension[layer]))*(dist[1]-dist[0]) + dist[0]

    # Parameter(Gewichtungen und Bias) initialisieren
    def init_parameters(self):
        parameter = []
        # von vorne nach hinten: in jedem Layer ...
        for i in range(0, len(self.distribution)):
            # dict: layer_parameters für Gewichtungen und Bias
            layer_parameters = {}
            for j in self.distribution[i].keys():
                # wenn es vordefinierte Gewichutngen in diesem Layer gibt, ...
                if j == 'b':
                    layer_parameters['b'] = self.init_parameters_b(i)
                    continue
                elif j == 'w':
                    # wenn es vordefinierte Bias in diesem Layer gibt, ...
                    layer_parameters['w'] = self.init_parameters_w(i)
                    continue
            parameter.append(layer_parameters)
        # global Variable: self.parameters
        self.parameters = parameter

    # Prüfen
    def print_parameter(self):
        a_1 = len(self.dimension)
        a_2 = len(self.parameters)
        a_3 = len(self.distribution)
        if a_1 == a_2 == a_3:
            print(f'Das MLP hat {a_1} layers。')
        # Inputlayer
        if self.parameters[0]:
            for j in self.parameters[0].keys():
                if j =='b':
                    print('Bias b im {}. Layer: {}'.format(1, self.parameters[0]['b'].shape))
        else:
            print('Das Inputlayer hat keine Bias und die Bilder werden direkt in den Inputlayer eingegeben.')
        # Hiddenlayer
        for i in range(1, a_1):
            for j in self.parameters[i].keys():
                if j =='w':
                    print('Gewichtungen w im {} Layer: {}。'.format(i+1, self.parameters[i]['w'].shape))
                    continue    
                elif j =='b':
                    print('Bias b im {}. Layer: {}'.format(i+1, self.parameters[i]['b'].shape))
                    continue  
        print('\n')

    """Normalisierung"""
    def gaussian(self, x):
        # x: 2D np.array/Matrix
        mean = np.mean(x)
        std = np.std(x)
        return (x-mean)/std

    def scale(self, x):
        std = np.std(x)
        return x/std

    def division(self, x):
        return x/255

    """2D One_Hot_Label"""
    def one_hot_label(self, label):
        # label: 1D np.array, int
        lab = np.zeros((label.size, 10))
        for i, row in enumerate(lab):
            # label[i] muss Integer sein, weil nur Integer als Positionindex verwendet werden kann.
            row[label[i]] = 1
        return lab

    def load_training_data(self):
        # 3D np.array: (60000, 28, 28)
        images = load_train_images()
        # 2D np.array: (60000, 784)
        # Für jedes Bild: 2D np.array(1, 784)
        prod = images.shape[1]*images.shape[2]
        images = images.reshape(-1, prod)

        # 1D np.array
        labels = load_train_labels()
        # 2D One_Hot_Label
        labels = self.one_hot_label(labels)
        return images, labels

    def load_test_data(self):
        # 3D np.array: (60000, 28, 28)
        images = load_test_images()
        # 2D np.array: (60000, 784)
        # Für jedes Bild: 2D np.array(1, 784)
        prod = images.shape[1]*images.shape[2]
        images = images.reshape(-1, prod)

        # 1D np.array
        labels = load_test_labels()
        # 2D One_Hot_Label
        labels = self.one_hot_label(labels)
        return images, labels
    
    '''Aktivierungsfunktionen'''
    def tanh(self, x):
        # x: 2D np.array
        return np.tanh(x)
    
    def d_tanh(self, data):
        # data: Output von tanh
        return 1-data**2
    """
    def d_tanh(data):
        return np.diag(1-(data.flatten())**2)
    """

    def softmax(self, x):
        # x: 2D np.array
        exp = np.exp(x - x.max())
        return exp/exp.sum()

    """
    def d_softmax(self, data):
        sm = self.softmax(data)
        # np.diag(输入: 列表或一维数组): 创建一个对角矩阵
        # np.outer(输入: 列表或一维数组, 输入: 列表或一维数组): 使两个输入做外积(sm矩阵乘法sm.T)
        # 但根据上述softmax()函数的定义, 需要输入一个numpy对象: 一维数组。
        # 也就是说, Softmax-Outputlayer的输入和输出都必须是numpy对象: 一维数组。
        return np.diag(sm)-np.outer(sm, sm)
    """

    def bypass(self, x):
        # für Inputlayer
        return x

    def relu(self, x):
        # x: 2D np.array/Matrix
        return np.maximum(0, x)
    
    def d_relu(self, x):
        # x: x: 2D np.array/Matrix; ReLU()激活函数的输入
        return np.where(x>0, 1, 0)

    def d_crossEntropy(self, output_softmax, true_label):
        # loss-function对Softmax-Outputlayer的input参数(input_softmax)的求导
        return output_softmax-true_label
    """
    def d_crossEntropy(input_softmax, true_label):
        # loss-function对Softmax-Outputlayer的input参数(input_softmax)的求导
        output_softmax = softmax(input_softmax)
        return output_softmax - true_label
    """

    def d_sigmoid(self, data):
        return data*(1-data)
    
    """Forward Propagation"""
    def FP(self, image, Parameters):
        # Inputlayer
        if Parameters[0]:
            for j in Parameters[0].keys():
                if j =='b': 
                    l_in = image + Parameters[0]['b']
                    l_out = self.tanh(l_in)
        else:
            l_in = image
            l_out = self.bypass(l_in)
        # Wenn Hiddenlayer vorhanden, ...
        if len(Parameters) > 2:
            for i in range(1, len(Parameters)-1):
                l_in = np.dot(l_out, Parameters[i]['w']) + Parameters[i]['b']
                l_out = self.relu(l_in)
        # Outputlayer
        l_in = np.dot(l_out, Parameters[-1]['w']) + Parameters[-1]['b']
        l_out = self.softmax(l_in)
        return l_out
    
    def Reverse(self, data):
        # data: Liste
        return list(reversed(data))
    
    """Backward Propagation"""
    def BP(self, image, true_label):
        """Forward Propagation"""
        """Input and Output Protokoll"""
        l_in_list = []
        l_out_list = []
        if self.parameters[0]:
            for j in self.parameters[0].keys():
                if j =='b': 
                    l_in = image + self.parameters[0]['b']
                    l_in_list.append(l_in)
                    l_out = self.tanh(l_in)
                    l_out_list.append(l_out)
        else:
            l_in = image
            l_in_list.append(l_in)
            l_out = self.bypass(l_in)
            l_out_list.append(l_out)
        if len(self.parameters) > 2:
            for i in range(1, len(self.parameters)-1):
                l_in = np.dot(l_out, self.parameters[i]['w']) + self.parameters[i]['b']
                l_in_list.append(l_in)
                l_out = self.relu(l_in)
                l_out_list.append(l_out)
        l_in = np.dot(l_out, self.parameters[-1]['w']) + self.parameters[-1]['b']
        l_in_list.append(l_in)
        l_out = self.softmax(l_in)
        l_out_list.append(l_out)
        
        """Backward Propagation"""
        delta = []
        # Outputlayer    
        error = self.d_crossEntropy(l_out_list[-1], true_label)
        layer_delta = {}
        layer_delta['b'] = error
        layer_delta['w'] = np.dot(l_out_list[-2].T, error)
        delta.append(layer_delta)
        temp_out = np.dot(error, self.parameters[-1]['w'].T)
        # Wenn Hiddenlayer vorhanden, ...
        if len(self.parameters) > 2:
            for i in range(len(self.parameters)-1, 1, -1):
                layer_delta = {}
                layer_delta['b'] = temp_out*self.d_relu(l_out_list[i-1])
                layer_delta['w'] = np.dot(l_out_list[i-2].T, temp_out*self.d_relu(l_out_list[i-1]))
                delta.append(layer_delta)
                temp_out = np.dot(temp_out*self.d_relu(l_out_list[i-1]), self.parameters[i-1]['w'].T)  
        # Inputlayer
        if self.parameters[0]:
            for j in self.parameters[0].keys():
                layer_delta = {}
                if j =='b': 
                    layer_delta['b'] = temp_out*self.d_tanh(l_out_list[0])
                    delta.append(layer_delta)
        else:
            delta.append({})
        # die Parameterliste umkehren, weil BP von hinten nach vorne arbeitet
        return self.Reverse(delta)

    """Parameterliste aktualisieren"""
    def combine_parameters(self, Parameters, learning_rate, delta):
        Parameters_temp = copy.deepcopy(Parameters)
        if self.parameters[0]:
            for j in self.parameters[0].keys():
                if j =='b': 
                    Parameters_temp[0]['b'] += learning_rate * (-1) * delta[0]['b']
        for i in range(1, len(self.parameters)):
            Parameters_temp[i]['b'] += learning_rate * (-1) * delta[i]['b']
            Parameters_temp[i]['w'] += learning_rate * (-1) * delta[i]['w']
        return Parameters_temp

    """Iteration"""
    def training_net(self, train_images, true_train_labels, learning_rate):
        temp_delta = []
        if self.parameters[0]:
            for j in self.parameters[0].keys():
                if j =='b':
                    temp_delta_0 = np.zeros((1, self.dimension[0]))
                    for j in range(train_images.shape[0]):
                        delta = self.BP(train_images[[j]], true_train_labels[[j]])
                        temp_delta_0 += delta[0]['b'] 
                    layer_delta_0 = {}
                    layer_delta_0['b'] = temp_delta_0/train_images.shape[0] # Mittlung der Gradienten aller Bildern (in einer Batch)
                    temp_delta.append(layer_delta_0)
        else:
            temp_delta.append({})
        for i in range(1, len(self.parameters)):
            temp_delta_b = np.zeros((1, self.dimension[i]))
            temp_delta_w = np.zeros((self.dimension[i-1], self.dimension[i]))
            for j in range(train_images.shape[0]):
                delta = self.BP(train_images[[j]], true_train_labels[[j]])
                temp_delta_b += delta[i]['b']
                temp_delta_w += delta[i]['w']
            layer_delta = {}
            layer_delta['b'] = temp_delta_b/train_images.shape[0] # Mittlung der Gradienten aller Bildern (in einer Batch)
            layer_delta['w'] = temp_delta_w/train_images.shape[0] # Mittlung der Gradienten aller Bildern (in einer Batch)
            temp_delta.append(layer_delta)
        return self.combine_parameters(self.parameters, learning_rate, temp_delta)

    """Prüfen"""
    def cross_Entropy(self, images, true_labels, Parameters):
        loss_fun_accu = 0
        for i in range(images.shape[0]):
            l_out = self.FP(images[[i]], Parameters)
            l_out_log = np.log(l_out)
            temp = (-1)*np.dot(true_labels[[i]], l_out_log.T)
            loss_fun_accu += temp
        return loss_fun_accu

    def test_accuracy(self, test_images, test_labels, Parameters):
        sum = 0
        for j in range(test_images.shape[0]):
            pred_label = np.argmax(self.FP(test_images[[j]], Parameters))
            true_label = np.argmax(test_labels[[j]])
            if pred_label == true_label:
                sum += 1
        print('Aus allen {} Bildern sind {} Bildern richtig erkannt.'.format(test_images.shape[0], sum))
        print(f'Accuracy: {sum/test_images.shape[0]}')

    """循环训练神经网络"""
    """mini batch learning as SGD: zufällige Auswahl von Trainingsbeispielen gleicher Menge mit Zurücklegen"""
    def training(self, train_images, true_train_labels, test_images, true_test_labels):
        for epoch in range(self.epoch//2):
            # 学习完每一个batch, 即向最优解/最优参数'走一步'。
            # 总共要'走(train_images.shape[0]//self.batch_size=)600步'。
            row_train_img = np.arange(train_images.shape[0])
            for i in range(train_images.shape[0]//self.batch_size):
                if i%100==99:
                    print('Running batch: {}/{}'.format(i+1, train_images.shape[0]//self.batch_size))
                # 打乱原先训练图集的行的序列, 并返回
                np.random.shuffle(row_train_img)
                # 在打乱的行的序列中抽取batch_size个训练图片, 用于训练神经网络。
                self.parameters = self.training_net(train_images[row_train_img[0:self.batch_size]], true_train_labels[row_train_img[0:self.batch_size]], 10**(-1.1))
                
            # 每一次迭代后测试神经网络数字识别的正确性。
            # loss_fun_accu_train = self.cross_Entropy(train_images, true_train_labels, self.parameters)
            # print('在第{}次迭代后, loss function在训练集中累加的结果为{}。'.format(epoch+1, loss_fun_accu_train))
            # loss_fun_accu_test = self.cross_Entropy(test_images, true_test_labels, self.parameters)
            # print('在第{}次迭代后, loss function在测试集中累加的结果为{}。'.format(epoch+1, loss_fun_accu_test))
            
            self.test_accuracy(train_images, true_train_labels, self.parameters)
            self.test_accuracy(test_images, true_test_labels, self.parameters)
            print('\n')

        # 返回训练完的参数列表
        return self.parameters

    """提取训练中被错误识别的图片并再次训练"""
    def find_same_element(self, data):
        output_list = []
        for i in range(len(data)-1):
            for k in range(i+1, len(data)):
                for j in data[i]:
                    if j in data[k] and j not in output_list:
                        output_list.append(j)
        return output_list

    def find_different_element(self, data):
        output_list = []
        for i in range(len(data)):
            for j in data[i]:
                if j not in self.find_same_element(data) and j not in output_list:
                    output_list.append(j)
        output_list.extend(self.find_same_element(data))
        return sorted(output_list)

    """选择合适的学习率learning_rate"""
    def find_lr(self, train_images, true_train_labels):
        loss_list = []
        lower = -2.5
        upper = -0.5
        step = 0.2
        # 挑选任意一组batch用于训练。
        rand_batch = np.random.randint(train_images.shape[0]//self.batch_size)
        # 用不同的learning_rate更新一次参数列表, 并计算出各自的损失函数的累加值。
        for lr_power in np.linspace(lower, upper, num=int((upper-lower)//step+1)):
            learning_rate = 10**lr_power
            temp_parameters = self.training_net(train_images[rand_batch*self.batch_size:(rand_batch+1)*self.batch_size], true_train_labels[rand_batch*self.batch_size:(rand_batch+1)*self.batch_size], learning_rate)
            loss_fun = self.cross_Entropy(train_images, true_train_labels, temp_parameters)
            loss_list.append([lr_power, loss_fun])
        plt.plot(np.array(loss_list)[:, 0], np.array(loss_list)[:, 1], color='black')
        plt.show()


# Modul testen
if __name__ == "__main__":

    """Struktur definieren, Gewichtungen und Bias initialisieren"""
    # ein Instanz ‘net’ aus Klasse 'Net' erstellen
    net = Net(Shape=[784, 100, 10], Distribution=[
                                                  {},
                                                  {'w': [-math.sqrt(6/(784+100)), math.sqrt(6/(784+100))], 'b': [0, 0.1]},
                                                  {'w': [-math.sqrt(6/(100+10)), math.sqrt(6/(100+10))], 'b': [0, 0.1]},
                                                 ], Batch_size=50)
    # alle Gewichtungen und Bias initialisieren und in einer Liste speichern
    net.init_parameters()
    # prüfen
    net.print_parameter()

    """调取MNIST中所有分别用来训练的图片和对应的标签"""
    temp_train_images, train_labels = net.load_training_data()
    train_images = net.gaussian(temp_train_images)
    temp_test_images, test_labels = net.load_test_data()
    test_images = net.gaussian(temp_test_images)

    """训练神经网络"""
    # 用所有的训练图片和对应的标签循环训练神经网络, 更新参数列表self.parameters并赋值给Parameters变量。
    Parameters = net.training(train_images, train_labels, test_images, test_labels)
