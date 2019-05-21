import cv2
from chainer import cuda, Variable, serializers
from net import *
import numpy as np
from PIL import Image, ImageFilter
import time

RUN_ON_GPU = True
CAMERA_ID = 0 # 0 for integrated cam, 1 for first external can ....
WIDTH=1
HEIGHT=1
PADDING = 50
MEDIAN_FILTER = 1
KEEP_COLORS = False

model = FastStyleNet()
# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

def _transform(in_image,loaded,m_path):
    if m_path == 'none':
        return in_image
    if not loaded:
        serializers.load_npz(m_path, model)
        if RUN_ON_GPU:
            cuda.get_device(0).use() #assuming only one core
            model.to_gpu()
        print ("loaded")

    xp = np if not RUN_ON_GPU else cuda.cupy
    
    image = np.asarray(in_image, dtype=np.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    if PADDING > 0:
        image = np.pad(image, [[0, 0], [0, 0], [PADDING, PADDING], [PADDING, PADDING]], 'symmetric')
    image = xp.asarray(image)
    x = Variable(image)
    y = model(x)
    result = cuda.to_cpu(y.data)
    if PADDING > 0:
        result = result[:, :, PADDING:-PADDING, PADDING:-PADDING]
    result = np.uint8(result[0].transpose((1, 2, 0)))
    med = Image.fromarray(result)
    if MEDIAN_FILTER > 0:
        med = med.filter(ImageFilter.MedianFilter(MEDIAN_FILTER))
    if KEEP_COLORS:
        med = original_colors(Image.fromarray(in_image), med)
    
    return np.asarray(med)

if __name__ == '__main__':
    cv2.namedWindow("style")
    vc = cv2.VideoCapture(CAMERA_ID)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)
    if vc.isOpened():
        rval, frame = vc.read()
        loaded = False
        mpath = 'models/edtaonisl.model'
    else:
        rval = False
    while rval:
        cv2.imshow("style", frame)
        rval, frame = vc.read()
        
        start = time.time()
        frame = cv2.resize( _transform(frame,loaded,mpath), (0,0), fx=1.0, fy=1.0)
        print(time.time() - start, 'sec')
        
        loaded=True
        key = cv2.waitKey(1)
        if key == 49: # 1
            mpath='models/edtaonisl.model'
            loaded=False
        if key == 50: # 2
            mpath='models/natasha-russu.model'
            loaded=False
        if key == 51: # 3
            mpath='models/kandinsky_e2_crop512.model'
            loaded=False
        if key == 52: # 4
            mpath='models/composition.model'
            loaded=False
        if key == 53: # 5
            mpath='models/scream-style.model'
            loaded=False
        if key == 54: # 6
            mpath='models/candy.model'
            loaded=False
        if key == 55: # 7
            mpath='models/kanagawa.model'
            loaded=False
        if key == 56: # 8
            mpath='models/fur.model'
            loaded=False
        if key == 57: # 9
            mpath='none'
            loaded=False
        if 'c' == chr(key & 0xFF):
            KEEP_COLORS = not KEEP_COLORS
        if 'q' == chr(key & 0xFF):
            break
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break            
    cv2.destroyWindow("preview")
