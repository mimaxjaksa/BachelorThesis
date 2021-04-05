from PIL import Image
import numpy as np
import os, os.path
import cv2 as cv
import random

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA): 
    """
       
        input:
            - image(3-dimensional array of uint8): image to be rescaled
            - width, height(int): dimension of the rescaled image
            
        functionality:
            - rescales the input image to the size given by height and width inputs, but keeps the ratio oh the input image
            
        output:
            - rescaled image
            
    """
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
        
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        
    resized = cv.resize(image, dim, interpolation = inter)
    
    return resized

def folder_tree(root):
    """
       
        input:
            - root(string): path to the root folder in which the folder tree will be created
            
        functionality:
            - creates a folder tree neded for images and features storing
            
        output:
            - no explicit output
            
    """
    
    os.chdir(root)
    
    dir_list = [
        "\\class0",
        "\\class0\\dataset2",
        "\\class0\\dataset2\\video1",
        "\\class0\\dataset2\\video2",
        "\\class0\\dataset3",
        "\\class0\\dataset3\\Far",
        "\\class0\\dataset3\\Left",
        "\\class0\\dataset3\\MiddleClose",
        "\\class0\\dataset3\\Right",
        "\\class0\\dataset4",
        "\\class1",
        "\\class1\\dataset1",
        "\\class1\\dataset1\\FR",
        "\\class1\\dataset1\\FSRS",
        "\\class1\\dataset3",
        "\\class1\\dataset3\\Far",
        "\\class1\\dataset3\\Left",
        "\\class1\\dataset3\\MiddleClose",
        "\\class1\\dataset3\\Right",
        "\\class1\\dataset4",
        ]
    
    os.mkdir(root + "\\data for extraction")
    
    for dir_ in dir_list:
        os.mkdir(root + "\\data for extraction" + dir_)
        
    os.mkdir(root + "\\features")
    
    for dir_ in dir_list:
        os.mkdir(root + "\\features" + dir_)

def get_random_crop(image, crop_height, crop_width):
    """
       
        input:
            - image(3-dimensional array of uint8): source image in witch random subimages are searched
            
        functionality:
            - return a random subimage of the input image
            
        output:
            - random subimage of size (crop_height, crop_width)
            
    """
    
    
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

def get_random_random_crop(image):
    sizes_list = [(200,150),(150,120),(100,80),(80,60),(50,40)]
    size = random.choice(sizes_list)
    
    return get_random_crop(image, size[1], size[0])

def pictures_from_dataset1(root, side = False):
    """
       
        input:
            - root(string): path to the root folder witch contains the folder tree containing uncropped images and labels for the bounding boxes
            
        functionality:
            - parses through a list of all images cropps them and resizes them to (100,100) according to the labels, 
                and puts all cars in Front(F) and Rear(R) in one folder and Front-Side(FS) and Rear-Side(RS) in another folder
                
        output:
            - no explicit output
            
    """
    
    output_root = root + "\\data for extraction\\dataset1"
    list_path = root + "\\dataset1\\train_test_split\\classification"
    label_path = root + "\\dataset1\\label"
    images_path = root + "\\dataset1\\image"
    
    i = 0
    i_ = 0
    for list_file in os.listdir(list_path):
        file = open(list_path + '\\' + list_file,'r')
        
        names = list(file)
        
        for name in names:
            name = name[:-1]
            label_name = name[:-3] + 'txt'
            full_image_path = images_path + "\\" + name
            full_label_path = label_path + "\\" + label_name
            
            
            current_image = Image.open(full_image_path, 'r')
            current_label = open(full_label_path)
                        
            position = int(current_label.readline())
            if (position == 1 or position == 2 ):
                num_cars = int(current_label.readline())
                for j in range(num_cars):
                    box = current_label.readline()
                    box = box[:-1]
                    box = [int(i) for i in box.split()]
                    im = current_image.copy()
                    im = im.crop(box)
                    im = im.resize((100,100), Image.ANTIALIAS)
                    destination_name = str(i) + ".jpg"
                    new_path = output_root + '\\FR\\'
                    full_destination_path = new_path + destination_name
                    im.save(full_destination_path)
                    i += 1
                    
            if side:
                if (position == 1 or position == 2 or position == 4 or position == 5):
                    num_cars = int(current_label.readline())
                    for j in range(num_cars):
                        box = current_label.readline()
                        box = box[:-1]
                        box = [int(i) for i in box.split()]
                        im = current_image.copy()
                        im = im.crop(box)
                        im = im.resize((100,100), Image.ANTIALIAS)
                        destination_name = str(i_) + ".jpg"
                        new_path = output_root + '\\FSRS'
                        full_destination_path = new_path + destination_name
                        im.save(full_destination_path)
                        i_ += 1
                
            
            current_label.close()
        
        
        file.close()
  
def pictures_from_dataset2(root):
    """
       
        input:
            - root(string): path to the root folder witch contains the folder containing the videos from witch the images for training are extracted
            
        functionality:
            - parses through frames of both videos, takes few random subimages from each frame and stores them all in two folders
            
        output:
            - no explicit output
            
    """
    
    video1_path = root + "\\data for extraction\\class0\\dataset2\\video1\\"
    video2_path = root + "\\data for extraction\\class0\\dataset2\\video2\\"
    
    video = cv.VideoCapture(root + "\\dataset2\\0005VD.mxf")
    i = 0
    while(i < 10000):
        ret, frame = video.read()
        if ret == False:
            break
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = image_resize(frame, width = 500)
        
        # for j in range(3):
        subim = get_random_random_crop(frame)
        subim = Image.fromarray(subim,'RGB')
        destination_name = str(i) + ".jpg"
        full_destination_path = video1_path + destination_name
        subim.save(full_destination_path)
        i += 1
            
    video = cv.VideoCapture(root + "\\dataset2\\0006R0.mxf")
    i = 0
    while(i < 10000):
        ret, frame = video.read()
        if ret == False:
            break
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = image_resize(frame, width = 500)
        
        # for j in range(3):
        subim = get_random_random_crop(frame)
        subim = Image.fromarray(subim,'RGB')
        destination_name = str(i) + ".jpg"
        full_destination_path = video2_path + destination_name
        subim.save(full_destination_path)
        i += 1
            
def pictures_from_dataset4(root):
    """
       
        input:
            - root(string): path to the root folder which contains folders with images 
            
        functionality:
            - parses through folders with images and takes random subimages and stores them in one folder
            
        output:
            - no explicit output
            
    """
    os.chdir(root)
    i = 0
    for file in os.listdir(root + '\\dataset4\\jaksa'):
        name = root + '\\dataset4\\jaksa\\' + file
        im = cv.imread(name)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = image_resize(im, width = 500)
        for j in range(30):
            subim = get_random_random_crop(im)
            subim = Image.fromarray(subim,'RGB')
            destination_name = str(i) + ".jpg"
            full_destination_path = root + "\\data for extraction\\class0\\dataset4\\" + destination_name
            subim.save(full_destination_path)
            i += 1
            
    i = 0
    for file in os.listdir(root + '\\dataset4\\test'):
        name = root + '\\dataset4\\test\\' + file
        im = cv.imread(name)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = image_resize(im, width = 500)
        for j in range(30):
            subim = get_random_random_crop(im)
            subim = Image.fromarray(subim,'RGB')
            destination_name = str(i) + ".jpg"
            full_destination_path = root + "\\data for extraction\\class0\\dataset4\\" + destination_name
            subim.save(full_destination_path)
            i += 1
        
if __name__ == '__main__':
    root = "D:\\work\\gerrit\\renesas-cms\\Bachelor_thesis\\renesas-cms"
    folder_tree(root)
    pictures_from_dataset1(root)
    pictures_from_dataset2(root)   
    pictures_from_dataset4(root)

        
        
    
        
        
    
    


















































