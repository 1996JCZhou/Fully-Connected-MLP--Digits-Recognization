# Fully-Connected Multi-Layer Perceptrons for Digits-Recognization

Welcome to my GitHub project, which marks the beginning of my journey in computer vision and reflects my passion for exploring the field. This project served as the foundation for my subsequent research in image classification.

To gain a deeper understanding of the optimization process in machine learning, including Forward-Propagation and Backward-Propagation, and how computers interpret images, I decided to construct a three-layer Multi-Layer Perceptrons (MLP) network from scratch, using only the Numpy library. This custom-built MLP network aims to accomplish the task of handwritten digit recognition. The project encompasses various critical tasks, such as designing the MLP network architecture, initializing weights and biases, implementing diverse normalization methods for different types of image data, defining activation functions (tanh, softmax, relu), and, most importantly, understanding the entire process of Forward-Propagation and Backward-Propagation for training the network.

In addition to the core implementation, I introduce PyQt5, a versatile toolkit, to design a user-friendly graphical user interface for seamless interactions. The PyQt5 integration serves two main purposes:
- After the completion of model training, the user can leverage the trained MLP network to predict handwritten digit images randomly extracted from the test dataset. This involves executing the Forward-Propagation process to obtain the final predictions for each digit.
- To facilitate custom user input, the project enables users to utilize the touchpad on their computers to draw handwritten digit images. By holding down the left key on the touchpad while moving their finger, users can create a handwritten digit, which the trained model then predicts, generating a numerical output.

The training dataset used to optimize the model parameters is sourced from the MNIST dataset. As for testing the model's performance, a portion of the test dataset comes from MNIST, while the other portion is comprised of user-input handwritten digit images drawn directly on the computer touchpad.

Through this project, I not only gained valuable insights into machine learning and computer vision but also honed my skills in building custom neural networks and creating interactive graphical interfaces.
