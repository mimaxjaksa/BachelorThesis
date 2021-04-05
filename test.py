from PIL import Image
import numpy as np
import os
import math as m
import cv2 as cv
import scipy.ndimage.filters as filters
import time

def simulate(root, hist):
    """
       
        input:
            - root(string): path to the root folder containing files of saved weights and biases of the trainied network
            - hist(array): input histogram to bre predicted 
            
        functionality:
            - simlates the forward pass of a neural network using files in root folder
            
        output:
            - float: in range(0,1) closer to 1 the more the model thinks input is a car
            
    """
    os.chdir(root)
    w01 = np.loadtxt('w01.txt')
    w12 = np.loadtxt('w12.txt')
    w23 = np.loadtxt('w23.txt')
    b1 = np.loadtxt('b1.txt')
    b2 = np.loadtxt('b2.txt')
    b3 = np.loadtxt('b3.txt')
    
    l1 = np.dot(hist, w01) + b1
    l1 = np.maximum(l1,0)
    l2 = np.dot(l1, w12) + b2
    l2 = np.maximum(l2,0)
    out = np.dot(l2, w23) + b3
    out = 1/(1 + m.exp(-out))
    
    return out

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

def get_hist(img,n = 64):
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
	
def split_in_bins(r,g,b,n = 64):
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

def non_max_suppression_fast(boxes, overlapThresh):
    """
       
        input:
            - boxes(list): list of boxes, each box contains 4 points
            - overlapTreshg(float): overlap ratio for box "merging" 
            
        functionality:
            - return only the boxes that are not more than overlapTresh part of another box
            
        output:
            - list of remaining boxes
            
    """
    if len(boxes) == 0:
       return []
    
    if boxes.dtype.kind == "i":
       boxes = boxes.astype("float")
     
    pick = []
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
    
       last = len(idxs) - 1
       i = idxs[last]
       pick.append(i)
    
    
       xx1 = np.maximum(x1[i], x1[idxs[:last]])
       yy1 = np.maximum(y1[i], y1[idxs[:last]])
       xx2 = np.minimum(x2[i], x2[idxs[:last]])
       yy2 = np.minimum(y2[i], y2[idxs[:last]])
    
       w = np.maximum(0, xx2 - xx1 + 1)
       h = np.maximum(0, yy2 - yy1 + 1)
    
       overlap = (w * h) / area[idxs[:last]]
    
       idxs = np.delete(idxs, np.concatenate(([last],
          np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

def image_search(root, im, n = 64):
    """
       
        input:
            - root(string): path to the root folder containing the weights and biases
                argument is passed to simulate() function
            - im(3-dimensional array of uint8): input image for search
            - n(int): 
            
        functionality:
            - resizes the input image to contant width with maintained aspect ratio, then searches through the image
                with several window sizes and puts all found bounding boxes in one list, after that the non-maximum supression\
                algorithm merges the boxes that are not necessary. It creates an image from input image with drawn rectangle for 
                remaining boxes and returns that image.
            
        output:
            - result image with drawn boxes 
            
    """
    
    d1 = 500
    r = np.shape(im)[1]/d1
    im = image_resize(im, width = d1)
    res_im = im
    
    search_size_list = [(200,150),(150,120),(100,80),(80,60),(50,40),(40,30)]
    
    boxes = np.zeros([1,4])
    # search_size_list = [(650,500)]
    for search_size in search_size_list:
        
        # search_size = (m.ceil(search_size[0]/m.floor(r)),m.ceil(search_size[1]/m.floor(r)))
        stride = 20
        threshold = 0.8
        range1 = list(range(int(np.shape(im)[0]/3), int((np.shape(im)[0] - search_size[1])*0.8), stride))
        range2 = list(range(0, np.shape(im)[1] - search_size[0], stride))
        res = np.zeros([len(range1), len(range2)])
        for i in range1:
            for j in range2:
                subimg = im[i:i + search_size[1], j:j + search_size[0], :]
                histogram = get_hist(subimg,n)
                result = simulate(root, histogram)
                res[range1.index(i),range2.index(j)] = result
                
                
            print(i)
            
        neighborhood_size = 9
        data = res
        
        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        maxima[res < threshold] = False    
        
        res = maxima
        
        for i in range(0,np.shape(res)[0]):
            for j in range(0,np.shape(res)[1]):
                if res[i,j]:
                    box_ = np.zeros([1,4])
                    box_[0,0] = range2[j]
                    box_[0,1] = range1[i]
                    box_[0,2] = range2[j] + search_size[0]
                    box_[0,3] = range1[i] + search_size[1]
                    
                    
                    
                    boxes = np.append(boxes, box_, axis = 0)
                                       
    boxes = np.delete(boxes, 0, axis = 0)
    boxes = non_max_suppression_fast(boxes, 0.5)
    
    
    for i in range(np.shape(boxes)[0]):
        box = boxes[i,:]
        res_im = cv.rectangle(im, (box[0], box[1]), (box[2],box[3]), (255,0,0), 2)


    # res_im = Image.fromarray(res_im)
    
    # end = time.time()
    # print(end - start)
    
    # ret

    return res_im
      
def video_from_folder(root, folder, video_name):
    """
       
        input:
            - root(string): path to the root folder in which the folder with images is
            - video_name(string): name of the video to be created in root folder
            
        functionality:
            - creates a video from a folder with images
            
        output:
            - no explicit output
            
    """
    os.chdir(root)
    image_folder = root + "\\" + folder
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = sorted(images, key=lambda x: int(x.split(".")[0]))    
    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv.VideoWriter(video_name, 0, 24, (width,height))
    
    for image in images:
        video.write(cv.imread(os.path.join(image_folder, image)))
    
    cv.destroyAllWindows()
    video.release()

def folder_from_video(root, video_name, folder_name):
    """
       
        input:
            - root(string): path to the root folder in which the video is
            - folder_name(string): name of the folder to be created in root folder
            
        functionality:
            - creates a folder from a freames of the video
            
        output:
            - no explicit output
            
    """
    os.chdir(root)
    input_video = cv.VideoCapture(root + video_name)
    
    i = 0
    while True:
        ret,frame = input_video.read()
        if ret == False:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        res_im = image_search(root, frame, n)
        res_im = Image.fromarray(res_im)
        res_im.save(root + "\\" + folder_name + "\\" +  str(i) + ".jpg")
        i += 1
        
    
def video_search(root, video_name, folder = "temp"):
    """
       
        input:
            - root(string): path to the root folder in which the video is
            - video_name(string): name of the video to be searched
            
        functionality:
            - creates a video from a freames of the input video, and saves it
            
        output:
            - no explicit output
            
    """
    os.mkdir(folder)
    folder_from_video(root, video_name, folder)
    res_name = video_name[:-4] + "_res.avi"
    video_from_folder(root, folder, res_name)
    os.rmdir(folder)
    
def folder_search(root, folder_name):
    """
       
        input:
            - root(string): path to the root folder in which the folder is
            - folder_name(string): name of the folder to be searched
            
        functionality:
            - creates a resulting folder from the input folder
            
        output:
            - no explicit output
            
    """
    os.chdir(root)
    res_name = folder_name + "_res"
    os.mkdir(res_name)
    for name in os.listdir(folder_name):
        file = root + "\\" + folder_name + "\\" + name
        im = cv.imread(file)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        res_im = image_search(root, im, n)
        res_im = Image.fromarray(res_im)
        res_im.save(res_name + "\\" + name)
 
def image_search_full(root, im_name):
    im = cv.imread(im_name)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    res_im = image_search(root, im)
    res_im = Image.fromarray(res_im)
    res_im.save(im_name[:-4] + "_res.jpg")
    
if __name__ == '__main__': 
    root = "D:\\work\\gerrit\\renesas-cms\\Bachelor_thesis\\renesas-cms"
    n = 64
    
    start = time.time()    
    
    # image_search_full(root, im_name = "im_test.jpg")
    video_search(root, video_name = "\\drugi.mkv", folder = "bla")
    video_search(root, video_name = "\\treci.mkv", folder = "bla1")
    video_search(root, video_name = "\\cetvrti.mkv", folder = "bla2")
    # folder_search(root, folder_name = "dataset4\\test")
    
    # video_from_folder(root, "temp", "blabbbb.avi")
    
    
    end = time.time()
    print(end - start)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    