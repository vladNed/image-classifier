import numpy as np

class Image_Classifier():

    def __init__(self):
        self.weights = np.load('data/coefs.npy')
        self.bias_vector = np.load('data/bias.npy')

    def softmax_function(self,X_array):
        """Softmax function for activation"""
        z_exp = [np.exp(i) for i in X_array]
        sum_z_exp = sum(z_exp)
        return [i / sum_z_exp for i in z_exp]


    def predict(self, X_test,):
        """Predict the class of a image."""
        prediction_array = self.softmax_function( np.dot(X_test,self.weights) + self.bias_vector )
        y_prediction = []

        for i in prediction_array:
            label = np.where(i == np.amax(i))[0][0]
            print(label)
            if label == 0:
                y_prediction.append('cat')
            if label == 1:
                y_prediction.append('dog')
            if label == 2:
                y_prediction.append('frog')
            if label == 3:
                y_prediction.append('horse')

        return np.array(y_prediction)

        