import tensorflow as tf
import detect_face #https://github.com/ShyBigBoy/face-detection-mtcnn
import cv2
import os,re
 
minsize = 40 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
gpu_memory_fraction=1.0
#创建网络和加载参数 
with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

#照片所在文件路径
path = 'D:/faceImages/wy/'

#获取整个文件图片路径
def getimgnames(path=path):
    filenames = os.listdir(path)
    imgnames = []
    for i in filenames:
        if re.findall('^\d+\.jpg$',i)!=[]:
            imgnames.append(i)
    return imgnames           

imgnames = getimgnames(path)
num = 1
for image_path in imgnames:
    img = cv2.imread(path+image_path)
    #得到人脸框          
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    #bounding_boxes中第一个为所需的人脸
    if bounding_boxes.shape[0]:
        face_position=bounding_boxes[0]
        face_position=face_position.astype(int)
        cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=img[face_position[1]:face_position[3],
                 face_position[0]:face_position[2],]
        #灰度化
        gary = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        #图片保存        
        cv2.imwrite("D:/faceImageGray/haha/{}.jpg".format(num), gary)
        num = num+1
    else:
        print('第',num,'张图无法识别')
        num = num+1
        
        