import pickle
import matplotlib.pyplot as plt
from softmax import *

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict

def gradient_ascent(model, target_class, init, learning_rate=1e-3):
    """
    Inputs:
    - model: Image classifier.
    - target_class: Integer, representing the target class the fooling image
      to be classified as.
    - init: Array, shape (1, Din), initial value of the fooling image.
    - learning_rate: A scalar for initial learning rate.
    
    Outputs:
    - image: Array, shape (1, Din), fooling images classified as target_class
      by model
    """
    
    image = init.copy()
    y = np.array([target_class])
    ###########################################################################
    # TODO: perform gradient ascent on your input image until your model      #
    # classifies it as the target class, get the gradient of loss with        #
    # respect to your input image by model.forwards_backwards(imgae, y, True) #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################   
    
    return image

def img_reshape(flat_img):
    # Use this function to reshape a CIFAR 10 image into the shape 32x32x3, 
    # this should be done when you want to show and save your image.
    return np.moveaxis(flat_img.reshape(3,32,32),0,-1)
    
    
def main():
    # Initialize your own model
    model = SoftmaxClassifier()
    config = {}
    target_class = None
    correct_image = None
    ###########################################################################
    # TODO: load your trained model, correctly classified image and set your  #
    # hyperparameters, choose a different label as your target class          #
    ###########################################################################    
    fooling_image = gradient_ascent(model, target_class, init=correct_image)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    

    
    ###########################################################################
    # TODO: compute the (magnified) difference of your original image and the #
    # fooling image, save all three images for your report                    #
    ###########################################################################

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


if __name__ == "__main__":
    main()
