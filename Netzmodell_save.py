"""MNN"""
import math
import pickle
from Net_algo import Net

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

"""Modelldatenspeicherung"""
with open('Save_Parameters.p', 'wb') as f:
    pickle.dump(Parameters, f)
