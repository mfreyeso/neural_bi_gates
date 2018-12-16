# Neural Binary Gates

**Neural Binary Gates** is a simple artificial neural network that represents
a problem related with digital binary gates. The operation that predict this
network is **X = (A xor B) and (A or C)**.

For resolve this problem I has designed a network with the foundations of
MLP (Multilayer Perceptron) and an architecture 3-3-1.

This network was built with Python 3.6 and Keras package. The network is represented
with *Dense* objects layers and use the optimizer SGD.

Finally for easy the use of this representation and test it I has deployed the
abstraction trained model on *Google Cloud ML Engine*, you can use the python
file *predict.py* for check the results. 
