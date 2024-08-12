import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

"""
Feedforward Model Object

    X -> Input Embedding -> Standardise -> Feedforward -> Standardise -> Output Decoding -> Yhat
                ^                               ^                               ^
               Bias                            Bias                            Bias
    
    Layers = 1 by default
    Row standardisation applied
    Output is non-thresholded activated values (confidence). No softmax applied due to multi-label output.
"""

def activation(input):
    return 1/(1+np.exp(-input))

class FeedForwardModel:
    def __init__(self, layers, subspaces, features, labels):
        """
        :param layers: Default and recommended setting = 1.
        :param subspaces: Value C, recommended = 20.
        :param features: Passed from main.py.
        :param labels: Passed from main.py.
        """
        self.input = None
        self.model_end = None
        self.layers, self.subspaces, self.features, self.labels = layers, subspaces, features, labels
        self.initialise_model(layers, subspaces, features, labels)
        return

    def initialise_model(self, layers, subspaces, features, labels):
        """
        Initialises model architecture only. Uses a link-list style object chain for matrix multiplication.
        Inference recursively calls next.forward() starting with input.forward() from FeedForwardModel.inference()
        E = encoder
            E.next()
                |
                L = block (feedforward layer)
                block.next() (assuming single layer)
                    |
                    D = decoder
                    D.next()
                        |
                        None (stub) - inference stops here and returns the output
        :param layers: Default and recommended setting = 1.
        :param subspaces: Value C, recommended = 20.
        :param features: Passed from main.py.
        :param labels: Passed from main.py.
        :return:
        """
        E, D, LayerEncoder, b = None, None, None, None
        encoder = Encoder(features, subspaces, E, b, self, layerID=">--- Architecture ---<\nEncoder at level 0")
        decoder = Decoder(labels, subspaces, D, b, layerID="Decoder at level {}".format(layers + 1))
        block = Block(subspaces, LayerEncoder, b, self,
                                                layerID="Block at level {}".format(1))
        self.add_layer(encoder, block)
        for i in range(layers - 1):
            block = Block(subspaces, LayerEncoder, b, self,
                                                    layerID="Block at level {}".format(i + 2))
            self.add_layer(self.model_end, block)
        self.add_layer(self.model_end, decoder)
        return

    def print(self):
        print(self.input.print())
        return

    def add_layer(self, block_prev, block_next):
        if self.input == None:
            self.input = block_prev
        block_prev.connect(block_next)
        self.model_end = block_next
        return

    def inference(self, X):
        return self.input.forward(X)

    def test_inference(self, X):
        return self.input.forward(X, test_inference = True)

    def set_weights(self,E, D, E_a, B_l, B_E, B_D):
        """
        :param E: Encoder weights (vector form).
        :param D: Decoder weights (vector form).
        :param E_a: Feedforward layer weights (vector form).
        :param B_l: Feedforward bias.
        :param B_E: Encoder bias.
        :param B_D: Decoder bias.
        :return:
        """
        weights = [E, D, E_a, B_l, B_E, B_D]
        self.input.update_weights(weights)
        return

class Encoder:
    def __init__(self, features, subspaces, E, bias, model, layerID):
        self.next = None
        self.E = E
        self.bias = bias
        self.features = features
        self.subspaces = subspaces
        self.layerID = layerID
        self.model = model
        return

    def print(self):
        model_summary = "" + str(self.layerID) +" "+str(self)+"\n"
        return self.next.print(model_summary)

    def connect(self, block_next):
        self.next = block_next
        return

    def forward(self, X, test_inference = False):
        """
        :param X: Input dataset.
        :return: Recursively call next to feedforward layer, passing encoded X.
        """
        out = np.add(np.matmul(X, self.E), self.bias.T)
        row_means = np.mean(out, axis=1, keepdims=True)
        row_stds = np.std(out, axis=1, keepdims=True)
        epsilon = 1e-10
        normout = (out - row_means) / (row_stds + epsilon)
        return self.next.forward(activation(normout), test_inference)

    def update_weights(self, weights):
        """
        :param weights: Weight values in array of vectors [E, D, E_a, B_l, B_E, B_D].
                        Reshape reorganises to matrix form.
        :return:
        """
        self.E = np.array(weights[0].reshape(self.features,self.subspaces))
        self.bias = np.array(weights[-2].reshape(self.subspaces,1))
        self.next.update_weights(weights,0)
        return

class Block:
    def __init__(self, subspaces, LayerEncoder, bias, model, layerID):
        self.next = None
        self.subspaces = subspaces
        self.LayerEncoder = LayerEncoder
        self.bias = bias
        self.layerID = layerID
        self.model = model
        return

    def print(self, model_summary):
        return self.next.print(model_summary + " " + str(self.layerID) +" "+str(self))+"\n"

    def connect(self, block_next):
        self.next = block_next
        return

    def forward(self, input, test_inference = False):
        """
        :param input:
        :return:
        """
        i = input.copy()
        out_pre_activation = np.add(np.matmul(i, \
                                              np.array(self.LayerEncoder).reshape(
                                                  (self.subspaces , self.subspaces))), \
                                    np.array(self.bias).reshape((1, self.subspaces)))
        row_means = np.mean(out_pre_activation, axis=1, keepdims=True)
        row_stds = np.std(out_pre_activation, axis=1, keepdims=True)
        epsilon = 1e-10
        out_pre_activation = (out_pre_activation - row_means) / (row_stds + epsilon)
        return self.next.forward(activation(out_pre_activation), test_inference)

    def update_weights(self, weights, layer):
        """
        :param weights: Weight values in array of vectors [E, D, E_a, B_l, B_E, B_D].
                        Reshape reorganises to matrix form.
        :param layer: Layer ID to index the feedforward layer. Default number of layers = 1.
                      E_a and B_l is an array of vectors (i.e., E_a = [E_a^1] for 1 feedforward layer).
        :return:
        """
        self.LayerEncoder = weights[2][layer]
        self.bias = weights[-3][layer]
        self.next.update_weights(weights,layer+1)
        return


class Decoder:
    def __init__(self, labels, subspaces, D, bias, layerID):
        self.labels, self.subspaces, self.D, self.bias = labels, subspaces, D, bias
        self.layerID = layerID
        self.next = None
        return

    def print(self, model_summary):
        return model_summary + str(self.layerID) +" "+str(self)+"\n"

    def forward(self, input, test_inference = False):
        """
        :param input:
        :return: activated label matrix
        """
        return activation(np.add(np.matmul(input, self.D), \
                                 self.bias.T))

    def update_weights(self, weights, layer):
        """
        :param weights: Weight values in array of vectors [E, D, E_a, B_l, B_E, B_D].
                        Reshape reorganises to matrix form.
        :return:
        """
        self.D = np.array(weights[1]).reshape((self.subspaces,self.labels))
        self.bias = np.array(weights[-1]).reshape((self.labels,1))
        return
