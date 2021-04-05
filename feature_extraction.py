import numpy as np
import os
import random
import cv2 as cv
from keras.models import Sequential
from keras.layers import Dense


def feature_extraction_from_folder(folder, n):   
    """
       
        input:
            - root(string): path to the folder containing images whose features need to be extracted
            
        functionality:
            - parses through every image, calculates histograms and appends them all in one variable
            
        output:
            - matrix: of size Nx48 where N is the number of pictures in input folder
            
    """
    data = np.zeros([1,3*n])
    
    for file in os.listdir(folder):
        name = folder + '\\' + file
        im = cv.imread(name)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        histogram = get_hist(im,n)
        data = np.append(data,histogram,axis = 0)       
        
    data = np.delete(data, 0, axis = 0)
    
    return data
    
def feature_extraction_all(root, n):
    
    """
   
        input:
            - root(string): path to the root folder containing folder tree of images of both classes
            
        functionality:
            - extracts features of all subfolders in folder \\data for extraction
                and stores those features in the same folder tree
            
        output:
            - no explicit output
        
    """
    data_root = root + "\\data for extraction"
    feature_root = root + "\\features"
    dir_list = [
        "\\class0\\dataset2\\video1",
        "\\class0\\dataset2\\video2",
        "\\class0\\dataset3\\Far",
        "\\class0\\dataset3\\Left",
        "\\class0\\dataset3\\MiddleClose",
        "\\class0\\dataset3\\Right",
        "\\class0\\dataset4",
        "\\class1\\dataset1\\FR",
        "\\class1\\dataset1\\FSRS",
        "\\class1\\dataset3\\Far",
        "\\class1\\dataset3\\Left",
        "\\class1\\dataset3\\MiddleClose",
        "\\class1\\dataset3\\Right",
        "\\class1\\dataset4",
        ]
    
    for dir_ in dir_list:
        data = feature_extraction_from_folder(data_root + dir_, n)
        os.chdir(feature_root + dir_)
        np.savetxt('data.txt', data)        
 
def group_features(root, n):
    """
       
        input:
            - root(string): path to the root folder containing folder tree of features to be grouped in one file
            
        functionality:
            - groups feature files into one file and saves it in the root folder to be used for training
            
        output:
            - no explicit output
            
    """
    
    feature_root = root + "\\features"
    dir_list_0 = [
        "\\class0\\dataset2\\video1",
        "\\class0\\dataset2\\video2",
        "\\class0\\dataset3\\Far",
        "\\class0\\dataset3\\Left",
        "\\class0\\dataset3\\MiddleClose",
        "\\class0\\dataset3\\Right",
        "\\class0\\dataset4"
        ]
    
    dir_list_1 = [
        "\\class1\\dataset1\\FR",
        # "\\class1\\dataset1\\FSRS", #model seems to be working better when not using Front-Side and Rear-Side pictures for traiing
        "\\class1\\dataset3\\Far",
        "\\class1\\dataset3\\Left",
        "\\class1\\dataset3\\MiddleClose",
        "\\class1\\dataset3\\Right",
        # "\\class1\\dataset4"
        ]
    
    class0 = np.zeros([1,3*n])
    class1 = np.zeros([1,3*n])
    
    for dir_ in dir_list_0:
        temp = np.loadtxt(feature_root + dir_ + "\\data.txt")
        class0 = np.append(class0, temp, axis = 0)
    
    for dir_ in dir_list_1:
        temp = np.loadtxt(feature_root + dir_ + "\\data.txt")
        class1 = np.append(class1, temp, axis = 0)
    
    class0 = np.delete(class0,0,axis = 0)
    class1 = np.delete(class1,0,axis = 0)
        
    os.chdir(root)
    np.savetxt('class0.txt',class0)
    np.savetxt('class1.txt',class1)
            
def split_in_bins(r,g,b,n = 16):
    """
       
        input:
            - r,b,g(array): 256 element histograms of R,G and B channels 
            - n(int): number of bins in which the histograms will be split
            
        functionality:
            - splits input histograms in bins by summing parts of the arrays together
            
        output:
            - array: of 3N elements, all three bin-split histograms concatenated together
            
    """
    r = list(r[0,:])
    g = list(g[0,:])
    b = list(b[0,:])
    h_ = []
    for x in range(0,len(r),int(len(r)/n)):
        h_.append(sum(r[x:x + n]))
    
    for x in range(0,len(g),int(len(r)/n)):
        h_.append(sum(g[x:x + n]))
        
    for x in range(0,len(b),int(len(r)/n)):
        h_.append(sum(b[x:x + n]))
    h_ = np.expand_dims(h_,0)
    return h_

def training(root, n, test = False):
    """
       
        input:
            - root(string): path to the folder containing extracted features for training
            
        functionality:
            - trains the neural network on grouped features of both classes
                if test = True split input and output data in train and test subsets and displays the accuracy.
                After training saves the weights and biases of the trained network in txt files so that they can be used 
                for simulating, but also then it can be used by other architectures
                
            
        output:
            - no explicit output
            
    """
    
    os.chdir(root)

    class0 = np.loadtxt('class0.txt')
    class1 = np.loadtxt('class1.txt')
    input_ = np.append(class0, class1, axis = 0)
    
    output_ = np.zeros([np.shape(class0)[0],1])
    output_ = np.append(output_, np.ones([np.shape(class1)[0],1]), axis = 0)

    
    c = list(zip(input_, output_))
    random.shuffle(c)    
    
    input_, output_ = zip(*c)
    num_train = np.shape(input_)[0]

    
    if (test):
        input_train = list(input_[:int(0.8*num_train)])
        output_train = list(output_[:int(0.8*num_train)])
        
        input_test = list(input_[int(0.8*num_train):])
        output_test = list(output_[int(0.8*num_train):])
        
        input_train = np.array(input_train)
        input_test = np.array(input_test)
        output_train = np.array(output_train)
        output_test = np.array(output_test)
    else:
        input_train = np.array(list(input_))
        output_train = np.array(list(output_))
    
    model = Sequential()
    model.add(Dense(20, input_dim = np.shape(input_)[1], activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense( 1, activation = 'sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(input_train, output_train, epochs = 10, batch_size = 128, shuffle = True)
    
    if (test):
        _, accuracy = model.evaluate(input_test, output_test)
        print('Accuracy: %.2f' % (accuracy*100))
    
    a = model.get_weights()
    np.savetxt('w01.txt',a[0])
    np.savetxt('b1.txt',a[1])
    np.savetxt('w12.txt',a[2])
    np.savetxt('b2.txt',a[3])
    np.savetxt('w23.txt',a[4])
    np.savetxt('b3.txt',a[5])
 
def get_hist(img,n = 16):
    """
       
        input:
            - img(3-dimensional array of uint8): input image for histogram calculation
            - n(int): number of bins in which the histogram will be split by the split_in_bins function
            
        functionality:
            - parses through the input image and generates a histogram of R,G and B channels, splits it in bins and normalizes it
            
        output:
            -(array of int of dimension 1xn) bin-split and concatenated histograms
            
    """
    dim1 = np.shape(img)[0]
    dim2 = np.shape(img)[1]
    r_ = img[:,:,0]
    g_ = img[:,:,1]
    b_ = img[:,:,2]
    
    r = np.zeros([1,256])
    g = np.zeros([1,256])
    b = np.zeros([1,256])
    
    for i in range(dim1):
        for j in range(dim2):
            p = r_[i,j]
            r[0,p] = r[0,p] + 1
        
            p = g_[i,j]
            g[0,p] = g[0,p] + 1
            
            p = b_[i,j]
            b[0,p] = b[0,p] + 1
    
    
    return split_in_bins(r,g,b,n)/dim1/dim2
    
if __name__ == '__main__': 
    root = "D:\\work\\gerrit\\renesas-cms\\Bachelor_thesis\\renesas-cms"
    n = 64
    feature_extraction_all(root, n)
    group_features(root, n)
    training(root, n)
    
    