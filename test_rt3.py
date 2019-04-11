from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

from SFD_pytorch import net_s3fd
from SFD_pytorch.bbox import *

from pathlib import Path
home = str(Path.home())


def detect(imgs):

    use_cuda = torch.cuda.is_available()
    model = os.path.join(home,"Dev/CyberCV/SFD_pytorch/data/s3fd_convert.pth")
    net = getattr(net_s3fd,"s3fd")()
    net.load_state_dict(torch.load(model))
    net.cuda()
    net.eval()

    img_results = []
    for img in imgs:
        img = img - np.array([104,117,123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,)+img.shape)

        img = Variable(torch.from_numpy(img).float())
        if use_cuda:
            img = img.cuda()
        BB,CC,HH,WW = img.size()
        with torch.no_grad():
            olist = net(img)

        #img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
        #BB,CC,HH,WW = img.size()
        #olist = net(img)

        bboxlist = []
        for i in range(int(len(olist)/2)): olist[i*2] = F.softmax(olist[i*2],dim=1)
        for i in range(int(len(olist)/2)):
            ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
            FB,FC,FH,FW = ocls.size() # feature map size
            stride = 2**(i+2)    # 4,8,16,32,64,128
            anchor = stride*4
            for Findex in range(FH*FW):
                windex,hindex = Findex%FW,Findex//FW
                axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
                score = ocls[0,1,hindex,windex]
                loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
                if score<0.05: continue
                priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
                variances = [0.1,0.2]
                box = decode(loc,priors,variances)
                x1,y1,x2,y2 = box[0]*1.0
                # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                if score >= 0.5:
                    bboxlist.append([x1,y1,x2,y2,score])
        bboxlist = np.array(bboxlist)
        keep = nms(bboxlist,0.3)
        bboxlist = bboxlist[keep,:]
        img_results.append(bboxlist.tolist())

        #if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return img_results

#parser = argparse.ArgumentParser(description='PyTorch face detect')
#parser.add_argument('--net','-n', default='s3fd', type=str)
#parser.add_argument('--model', default='', type=str)
#parser.add_argument('--path', default='CAMERA', type=str)

#args = parser.parse_args()


'''
if args.path=='CAMERA': cap = cv2.VideoCapture(0)
while(True):
    if args.path=='CAMERA': ret, img = cap.read()
    else: img = cv2.imread(args.path)

    imgshow = np.copy(img)
    bboxlist = detect(net,img)

    keep = nms(bboxlist,0.3)
    bboxlist = bboxlist[keep,:]
    for b in bboxlist:
        x1,y1,x2,y2,s = b
        if s<0.5: continue
        cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
    #cv2.imshow('test',imgshow)

    if args.path=='CAMERA':
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    else:
        cv2.imwrite(args.path[:-4]+'_output.png',imgshow)
        break
        #if cv2.waitKey(0) or True: break
'''
