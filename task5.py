import numpy as np
import tensorflow as tf
import detect_face #https://github.com/ShyBigBoy/face-detection-mtcnn
import cv2

cap = cv2.VideoCapture(0)#创建一个 VideoCapture 对象

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


#会话读入
sess = tf.Session()
saver = tf.train.import_meta_graph('model/train_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('model/'))
graph = tf.get_default_graph()
#with tf.Session() as sess:
#    t=sess.run(graph)
x_new = graph.get_tensor_by_name('x_data:0')
y_new = graph.get_tensor_by_name('y:0')
Name = ['daiyejun','gaohongbin','heziwei','lgh','ljh','lzy','pcm','renhuikang','wangjianwei','wy']

while(cap.isOpened()):#循环读取每一帧
    #返回两个参数，第一个是bool是否正常打开，第二个是照片数组，如果只设置一个则变成一个tumple包含bool和图片
    ret_flag, Vshow = cap.read() 
    #得到人脸框          
    bounding_boxes, _ = detect_face.detect_face(Vshow, minsize, pnet, rnet, onet, threshold, factor)

    #bounding_boxes中第一个为所需的人脸
    if bounding_boxes.shape[0]:# and bounding_boxes[0][0]>0 and bounding_boxes[0][1]>0:
        face_position=bounding_boxes[0]
        face_position=face_position.astype(int)
        cv2.rectangle(Vshow, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        crop=Vshow[face_position[1]:face_position[3], face_position[0]:face_position[2],]
        
        gary = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        da_new = cv2.resize(crop,(32,32))
        da_new = da_new/255
        data = np.reshape(da_new,[1,32,32,3])
        
        pre = sess.run(y_new,feed_dict={x_new:data})
        acc = np.max(pre)
        num = np.argmax(pre)   #循序
        font = cv2.FONT_HERSHEY_SIMPLEX  #字体
        if acc>0.8:
            cv2.putText(Vshow, Name[num]+'%.4f'%acc, (50, 300), font, 1.2, (255, 255, 255), 2)
        else:
            cv2.putText(Vshow, 'unknow', (50, 300), font, 1.2, (255, 255, 255), 2)
    cv2.imshow("Capture_Test",Vshow)  #窗口显示，显示名为 Capture_Test

    k = cv2.waitKey(1) & 0xFF #每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
    if k == 27: #若检测到按键 ‘esc’，退出
        break
cap.release() #释放摄像头
sess.close()
cv2.destroyAllWindows()#删除建立的全部窗口