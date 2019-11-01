############################################################################
#                              Cv.hw1                                      #  
#                        Arthor: Wet-ting Cao.                             #   
#                             2019.10.24                                   #
############################################################################

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsView, QGraphicsScene
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from opencv import Ui_MainWindow
import cv2 as cv
import numpy as np  

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math
from scipy import signal, misc

# MainWindow -> button implementation.
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # Q1.
        self.loadImage.clicked.connect(self.load)
        self.colorConv.clicked.connect(self.color)
        self.imgFlipping.clicked.connect(self.flip)
        self.blending.clicked.connect(self.blend)
        
        # Q2.
        self.globalThr.clicked.connect(self.glob)
        self.localThr.clicked.connect(self.local)
        
        # Q3.
        self.RST.clicked.connect(self.rst)
        self.PT.clicked.connect(self.pt)
        
        # Q4.
        self.gaussian.clicked.connect(self.gaussianFilter)
        self.sobelX.clicked.connect(self.sobelXFilter)
        self.sobelY.clicked.connect(self.sobelYFilter)
        self.magnitude.clicked.connect(self.magnitudeFilter)
        
        # Q5.
        self.Imgs.clicked.connect(self.showImgs)
        self.Hyper.clicked.connect(self.showHyper)
        self.Train.clicked.connect(self.train)
        self.Result.clicked.connect(self.showResult)
        self.Inf.clicked.connect(self.showInf)
        
    # 1.1 Load Image File 
    def load(self) :
        path = 'images/images/dog.bmp'
        img = mpimg.imread(path)
        h, w = img.shape[:2]
        print('Height = ' + str(h))
        print('Width = ' + str(w))
        
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        
    # 1.2 Color Conversion  
    def color(self) :
        path = 'images/images/color.png'
        img = mpimg.imread(path)
        conv_img = img[:, :, (2, 0, 1)]
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(img), plt.title('original')
        plt.subplot(122), plt.axis('off'), plt.imshow(conv_img), plt.title('Convention')
        plt.show() 
    
    # 1.3 Image Flipping
    def flip(self) :
        path = 'images/images/dog.bmp'
        img = mpimg.imread(path)
        flip = cv.flip(img, 1)
        
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(img), plt.title('original')
        plt.subplot(122), plt.axis('off'), plt.imshow(flip), plt.title('Flipping')
        plt.show()
        
    # 1.4 Blending 
    def blend(self) : 
    
        def nothing(x):
            pass
            
        path = 'images/images/dog.bmp'
        img = mpimg.imread(path)
        flip = cv.flip(img, 1)
        
        output = cv.addWeighted(img, 1, flip, 0, 0)
        
        winName = 'Blending'
        trackName = 'Blend: '
        cv.namedWindow(winName)
        cv.createTrackbar(trackName, winName, 0, 100, nothing)
        print('Please enter "esc" to exit blending window.')
        
        while(1):
        
            cv.imshow(winName, cv.cvtColor(output, cv.COLOR_BGR2RGB))
            
            if cv.waitKey(33) == 27:
                break
                
            alpha = cv.getTrackbarPos(trackName, winName) / 100
            beta = 1 - alpha            
            output = cv.addWeighted(img, alpha, flip, beta, 0)
        
        cv.destroyAllWindows()    
    
    # 2.1 Global Threshold
    def glob(self) :
        path = 'images/images/QR.png'
        img = cv.imread(path, 0)
        ret, th1 = cv.threshold(img, 80, 255, cv.THRESH_BINARY)
        
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(img, 'gray'), plt.title('original')
        plt.subplot(122), plt.axis('off'), plt.imshow(th1, 'gray'), plt.title('global threshold')
        plt.show()
        
    # 2.2 Local Threshold
    def local(self) :
        path = 'images/images/QR.png'
        img = cv.imread(path, 0)
        th1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 19, -1)
        
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(img, 'gray'), plt.title('original')
        plt.subplot(122), plt.axis('off'), plt.imshow(th1, 'gray'), plt.title('global threshold')
        plt.show()
    
    # 3.1 Transforms: Rotation, Scaling, Translation
    def rst(self) :
        print('Transforms: Rotation, Scaling, Translation')       
        def translate(img, x, y):
            M = np.float32([[1, 0, x], [0, 1, y]])
            shifted = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
            return shifted
            
        def rotate(img, angle, scale = 1.0):
            ret, thresh = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 127, 255, 0)
            _, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            M = cv.moments(contours[0])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            h, w = img.shape[:2]
            center = (cx - 2, cy - 3)
            
            M = cv.getRotationMatrix2D(center, angle, scale)
            rotated = cv.warpAffine(img, M, (w, h))
            
            return rotated
            
        img = cv.imread('images/images/OriginalTransform.png')
        angle = self.angle.toPlainText()
        scale = self.scale.toPlainText()
        tx = self.tx.toPlainText()
        ty = self.ty.toPlainText()
        
        translated = translate(img, float(tx), float(ty))
        rotated = rotate(translated, float(angle), float(scale))
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('original')
        plt.subplot(122), plt.axis('off'), plt.imshow(cv.cvtColor(rotated, cv.COLOR_BGR2RGB)), plt.title('Tansforms')
        plt.show()  

    # 3.2 Perspective Transformation
    def pt(self) :
        global x0, y0, pt
        pt = []
        print('Perspective transform')
        def getPoint(event, x, y, flags, param):
            global x0, y0, pt
            if event == cv.EVENT_LBUTTONDOWN:
                x0, y0 = x, y
                pt.append([x0, y0])
           
        img = cv.imread('images/images/OriginalPerspective.png')
        cv.namedWindow('img')
        cv.setMouseCallback('img', getPoint)
        while (1):
            k = cv.waitKey(33)
            cv.imshow('img', img)
            if len(pt) == 4: 
                break
                
        print('pt1: (' + str(pt[0][0]) + ', ' + str(pt[0][1]) + ')')
        print('pt2: (' + str(pt[1][0]) + ', ' + str(pt[1][1]) + ')')
        print('pt3: (' + str(pt[2][0]) + ', ' + str(pt[2][1]) + ')')      
        print('pt4: (' + str(pt[3][0]) + ', ' + str(pt[3][1]) + ')')
        
        cv.destroyWindow('img')
        
        pts1 = np.float32([pt[0], pt[1], pt[3], pt[2]])
        pts2 = np.float32([[20, 20], [450, 20], [20, 450], [450,450]])
        
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        result = cv.warpPerspective(img, matrix, (450, 450))
        
        cv.imshow('Result', result)
        print('Enter any keys to close this window.')
        cv.waitKey(0)
        cv.destroyWindow('Result')
        
    # 4.1 gaussian
    def gaussianFilter(self) :
        path = 'images/images/School.jpg'
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_k = np.exp(-0.1 * (x ** 2 + y ** 2))
        
        gaussian_k /= gaussian_k.sum()
        #print(gaussian_k)
        img = mpimg.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(gray, cmap = plt.get_cmap('gray')), plt.title('to_gray')
        
        gaus = signal.convolve2d(gray, gaussian_k, boundary = 'symm', mode = 'same')
        plt.subplot(122), plt.axis('off'), plt.imshow(gaus, cmap = plt.get_cmap('gray')), plt.title('gaussian blur')
        plt.show()
        
        
    def sobelXFilter(self) :
        path = 'images/images/School.jpg'
        img = cv.imread(path)
        im = cv.imread(path, 0)

        sX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
        imgX = cv.filter2D(im, -1, sX)
        
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('original')      
        plt.subplot(122), plt.axis('off'), plt.imshow(imgX.astype('uint8'), cmap=plt.cm.gray), plt.title('sobelX filter')
        plt.show()
    
    def sobelYFilter(self) :
        path = 'images/images/School.jpg'
        img = cv.imread(path)
        im = cv.imread(path, 0)

        sY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
        imgY = cv.filter2D(im, -1, sY)
        
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('original')      
        plt.subplot(122), plt.axis('off'), plt.imshow(imgY.astype('uint8'), cmap=plt.cm.gray), plt.title('sobelY filter')
        plt.show()
    
    def magnitudeFilter(self) :
        path = 'images/images/School.jpg'
        img = cv.imread(path)
        im = cv.imread(path, 0)
        
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_k = np.exp(-0.1 * (x ** 2 + y ** 2))
        
        gaussian_k /= gaussian_k.sum()
        gaus = signal.convolve2d(im, gaussian_k, boundary = 'symm', mode = 'same')
        
        sX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
        sY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
        imgX = cv.filter2D(gaus, -1, sX)
        imgY = cv.filter2D(gaus, -1, sY)
        imgC = np.hypot(imgX, imgY)
        imgC /= np.max(imgC) / 255
        
        plt.figure(figsize = (15, 5))
        plt.subplot(121), plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('original')      
        plt.subplot(122), plt.axis('off'), plt.imshow(imgC.astype('uint8'), cmap=plt.cm.gray), plt.title('magnitude')
        plt.show()

        
    # 5.1 show training imgs.
    def showImgs(self) :
    
        def imshow(img, label, i):         
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()              
            plt.subplot(1, 10, i + 1)
            plt.title(classes[label])
            plt.axis('off')
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            
        print('Show training images.')
        # get some random training images
        dataiter = iter(show)
        images, labels = dataiter.next()

        # show images
        for i in range(10):
            imshow(torchvision.utils.make_grid(images[i]), labels[i], i)
        plt.show()
        
    # 5.2 show Hyperparams.
    def showHyper(self) :
        print('Show hyperparameters!')
        print('batch size: ' + str(batch))
        print('learning rate: ' + str(lr))
        print('optimizer: ' + optimizer)
        
    # 5.3 train 1 epoch.
    def train(self) :
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Lenet5().to(device)
        criteon = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr = lr)
        loss_iter = []
        iter = 0
        
        print('Start to train 1 epoch...')
        for epoch in range(1):
            model.train()
            for i, (x, label) in enumerate(trainloader) :
                x, label = x.to(device), label.to(device)
                y = model(x)
                loss = criteon(y, label)
                loss_iter.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter += 1
                
            plt.figure()
            plt.plot(loss_iter)
            plt.title('model loss (1 epoch): ' + str((sum(loss_iter).item() / iter)))
            plt.ylabel('loss'), plt.xlabel('iteration')
            plt.legend(['training loss'], loc = 'upper left')
            plt.show()
            
        print('Training loss at the end of the epoch is: ' + str((sum(loss_iter).item() / iter)))
    
    # 5.4 Show training result.
    def showResult(self) :
        acc = mpimg.imread('acc.png')
        loss = mpimg.imread('loss.png')
        
        plt.figure(figsize = (10, 12))
        plt.subplot(2, 1, 1)
        plt.axis('off')
        plt.imshow(acc)
        plt.subplot(2, 1, 2)
        plt.axis('off')
        plt.imshow(loss)
        plt.show()
        
    # 5.5 Inference
    def showInf(self) :    
        def imshow(img):         
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.figure(figsize = (15, 10))
            plt.subplot(1, 2, 1)
            plt.title('Show image')
            plt.axis('off')
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Lenet5().to(device)
        pretrained = torch.load('mnist_lenet5.pth')
        model.load_state_dict(pretrained)
        model.eval()
        num = self.Test_idx.toPlainText()
        print('You choose: ' + num)
        num = int(num)
        
        group = math.floor(num / batch)
        num %= batch
                
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        for _ in range(group):
            images, labels = dataiter.next()
        
        imshow(torchvision.utils.make_grid(images[num]))
        
        with torch.no_grad():
            prob = []
            x, label = images[num].to(device), labels[num].to(device)
            x = x.unsqueeze(0)
            y = model(x)
            y = F.softmax(y, dim = 1).cpu()
            for i in range(10):
                prob.append(y[0][i].item())
            plt.subplot(1, 2, 2)
            plt.bar(classes, prob)
            plt.title('Estimation result')
            plt.ylabel('prob'), plt.xlabel('classes')
            plt.show()
                    
############################################################################
#                                                                          #
#              Q5. train a Cifar-10 classifier using Lenet-5               #        
#                                                                          #
############################################################################

batch = 128
lr = 0.01
optimizer = 'SGD'
    
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

show = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=False, transform=transform)
show = torch.utils.data.DataLoader(show, batch_size=10,
                                          shuffle=True, num_workers=2)  
                                         
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=2)                                          

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        
        # (1, 32, 32) -> (6, 14, 14) -> (16, 5, 5)
        self.conv = nn.Sequential( 
            nn.Conv2d(1, 6, 5, 1, 2), # inch, outch, kernel, stride, padding.
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),  # kernel, stride, padding.
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv(x)
        x = x.view(batchsize, 16 * 5 * 5)
        x = self.fc(x) # logits
        return x

############################################################################
#                                                                          #
#        end   Q5. train a Cifar-10 classifier using Lenet-5               #        
#                                                                          #
############################################################################

                              
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
