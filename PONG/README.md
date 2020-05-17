Weights are stored in matrices.

Layer 1 of our Neural Network is a 200 x 6400 matrix representing the weights for our hidden layer.
For layer 1, element w1_ij represents the weight of neuron i for input pixel j in layer 1.

Layer 2 is a 200 x 1 matrix representing the weights of the output of the hidden layer on our final output.
For layer 2, element w2_i represents the weights we place on the activation of neuron i in the hidden layer.

We initialize each layer’s weights with random numbers for now.
Divide by the square root of the number of the dimension size to normalize our weights.

We use RMSprop, which is unpublished optimization algorithm designed for neural networks, first proposed by Geoff Hinton.
RMSprop is derived from Rprop.
Rprop combines the idea of only using the sign of the gradient with the idea of adapting the step size individually for each weight.
The central idea of RMSprop is keep the moving average of the squared gradients for each weight.
And then we divide the gradient by square root the mean square. Which is why it’s called RMSprop(root mean square)
