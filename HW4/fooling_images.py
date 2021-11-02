import pickle
import matplotlib.pyplot as plt
from softmax import *

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict

def gradient_ascent(model, target_class, init, learning_rate=1):
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
    iters = 20
    image = init.copy()
    print(image)
    mean_image = np.mean(image)
    std = np.std(image)
    image = (image - mean_image)/std
    
    
    y = np.array([target_class])
    ###########################################################################
    # TODO: perform gradient ascent on your input image until your model      #
    # classifies it as the target class, get the gradient of loss with        #
    # respect to your input image by model.forwards_backwards(imgae, y, True) #
    ###########################################################################
    
    print(model.forwards_backwards(image).argmax(axis=1) )
    for i in range(iters):
        g = model.forwards_backwards(image, y, True)
        image = image - g * learning_rate
        print(model.forwards_backwards(image).argmax(axis=1) )
        if model.forwards_backwards(image).argmax(axis=1) == y:
            print("Found adversarial example!")
            break
    image = image * std + mean_image
    image =  image.astype(int)
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
    model = SoftmaxClassifier(input_dim=3072, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0)
    config = {}
    target_class = 1
    batch1 = unpickle("cifar-10-batches-py/data_batch_1")
    images = batch1['data']
    labels = batch1['labels']
    print(images[1].shape, labels[1])
    correct_image = images[1]
    correct_image = correct_image.reshape([1,3072])
    ###########################################################################
    # TODO: load your trained model, correctly classified image and set your  #
    # hyperparameters, choose a different label as your target class          #
    ###########################################################################    
    model.load("model_trained")
    fooling_image = gradient_ascent(model, target_class, init=correct_image)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    

    
    ###########################################################################
    # TODO: compute the (magnified) difference of your original image and the #
    # fooling image, save all three images for your report                    #
    ###########################################################################
    print(np.linalg.norm(correct_image - fooling_image))

    difference = (correct_image - fooling_image).astype(int)
    
    print(difference)
    plt.imshow(correct_image.reshape([3,32,32]).swapaxes(0,2))
    plt.savefig("correct") # frog
    plt.close()
    print(fooling_image)
    plt.imshow(fooling_image.reshape([3,32,32]).swapaxes(0,2))
    plt.savefig("fool") # frog
    plt.close()
    plt.imshow(np.abs(difference).reshape([3,32,32]).swapaxes(0,2))
    plt.savefig("diff") # frog
    plt.close()
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


if __name__ == "__main__":
    main()
