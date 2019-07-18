import numpy as np
import tensorflow as tf

tf.reset_default_graph()   #清除变量

#数据读入
data_train = np.float32(np.load("data/data_train.npy"))
labels_train = np.load("data/labels_train.npy")
data_test = np.float32(np.load("data/data_test.npy"))
labels_test = np.load("data/labels_test.npy")

#独热化
labels_train,labels_test = tf.one_hot(labels_train,10),tf.one_hot(labels_test,10) #

x_data = tf.placeholder(tf.float32,[None,32,32,3],name='x_data')
y_data = tf.placeholder(tf.float32,[None,10])

#====数据卷积--池化--卷积--池化====
w1 = tf.Variable(tf.random_normal([3,3,3,32],stddev=0.01))  #卷积核/filter的初始权值
w2 = tf.Variable(tf.random_normal([3,3,32,50],stddev=0.01))  #卷积核/filter的初始权值

conv1 = tf.nn.conv2d(x_data,w1,strides=[1,1,1,1],padding='SAME')    #卷积
conv1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')   #池化
conv2 = tf.nn.conv2d(pool1,w2,strides=[1,1,1,1],padding='VALID')    #卷积
conv2 = tf.nn.relu(conv2)
pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')   #池化
data = tf.reshape(pool2,[-1,2450]) 
      
#=====BP神经网络=====
global_step = tf.Variable(0, trainable=False)    #迭代的速率变化
learning_rate = tf.train.exponential_decay(0.3, global_step, 100, 0.95)

w3 = tf.Variable (tf.random_normal ([2450,60], stddev = 0.1))  #隐层权值
bias1 = tf.Variable (tf.constant (0.1), [60])   #隐层阈值／偏置项
w4 = tf.Variable (tf.random_normal ([60,10], stddev = 0.1))  #输出层权值
bias2 = tf.Variable (tf.constant (0.1), [10])   #输出层阈值／偏置项

#输入层到隐层
H = tf.sigmoid(tf.matmul(data,w3)+bias1)
H = tf.nn.relu (H)


#隐层到输出层
y = tf.nn.softmax(tf.matmul(H,w4)+bias2,name='y')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y),axis=1))  #交叉熵
optimizer = tf.train.GradientDescentOptimizer(learning_rate)   #梯度下降法优化器
train = optimizer.minimize(cross_entropy,global_step=global_step)            #利用优化器对交叉熵进行优化
saver = tf.train.Saver()  #保存
init = tf.global_variables_initializer()
#===运行会话====
x_s =np.zeros([400,32,32,3],dtype='float32')
y_s =np.zeros([400,10],dtype='float32')
with tf.Session() as sess:
    sess.run(init)
    labels_train,labels_test = sess.run([labels_train,labels_test])
    for i in range(2001):
        k=0
        if i%100==0:  #每训练50轮打印一次训练集样本的预测精度
            pre = tf.equal(tf.argmax(y,axis=1),tf.argmax(y_data,axis=1))
            acc = sess.run(pre,feed_dict = {x_data: data_test, y_data: labels_test})
            print(i,'acc: ',sum(acc)/len(acc),sess.run(learning_rate))
            print(sess.run(cross_entropy,feed_dict = {x_data: x_s, y_data: y_s}))#交叉熵
        for j in np.random.randint(0,4799,size=[400]):
            x_s[k] = data_train[j,:]
            y_s[k] = labels_train[j]
            k = k+1
        sess.run(train,feed_dict = {x_data: x_s, y_data: y_s})
        
    saver.save(sess,'model/train_model')
#准确率为99.5%  




#    for i in range(4800):
#        pool_[i]=sess.run(pool2,feed_dict={img:data_train[i,:,:,:]})
#    for i in range(1200):
#        pool_test[i] = sess.run(pool2,feed_dict={img:data_test[i,:,:,:]})

##重塑数据结构
#data = np.reshape(pool_,[4800,7*7*50])
#data = data*10  #训练
#data_te = np.reshape(pool_test,[1200,7*7*50])
#data_te = data_te*10  #测试




##data_n = tf.nn.relu (data)
##=====BP神经网络=====
#labels_train,labels_test = tf.one_hot(labels_train,10),tf.one_hot(labels_test,10) #
##迭代的速率变化
#global_step = tf.Variable(0, trainable=False)    
#learning_rate = tf.train.exponential_decay(0.25, global_step, 200, 0.98)
#
##w3 = tf.Variable(tf.zeros([2450,60]))  #隐层权值
##w4 = tf.Variable(tf.zeros([60,10]))  #输出层权值
##bias1 = tf.Variable(tf.zeros([60]))   #隐层阈值／偏置项
##bias2 = tf.Variable(tf.zeros([10]))   #输出层阈值／偏置项
#w3 = tf.Variable (tf.random_normal ([2450,128], stddev = 0.1))  #隐层权值
#bias1 = tf.Variable (tf.constant (0.1), [128])   #隐层阈值／偏置项
#w4 = tf.Variable (tf.random_normal ([128,10], stddev = 0.1))  #输出层权值
#bias2 = tf.Variable (tf.constant (0.1), [10])   #输出层阈值／偏置项
#
#x_data = tf.placeholder(tf.float32,[None,2450],name='x_data')
#y_data = tf.placeholder(tf.float32,[None,10])
#
##输入层到隐层
#H = tf.sigmoid(tf.matmul(x_data,w3)+bias1)
#H = tf.nn.relu (H)
##隐层到输出层
#y = tf.nn.softmax(tf.matmul(H,w4)+bias2)
#
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y),axis=1))  #交叉熵
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)   #梯度下降法优化器
#train = optimizer.minimize(cross_entropy,global_step=global_step)            #利用优化器对交叉熵进行优化
#init = tf.global_variables_initializer()
##===构建会话====
#x_s =np.zeros([100,2450],dtype='float32')
#y_s =np.zeros([100,10],dtype='float32')

#with tf.Session() as sess:
#    sess.run(init)
#    labels_trai,labels_tes = sess.run([labels_train,labels_test])
#    for i in range(15000):
#        k = 0
#        if (i+1)%50==0:  #每训练50轮打印一次训练集样本的预测精度
#            pre = tf.equal(tf.argmax(y,axis=1),tf.argmax(labels_train,axis=1))
#            acc = sess.run(pre,feed_dict={x_data:data})
#            print(i,'训练: ',sum(acc)/len(acc),sess.run(learning_rate))
#            
#            pre = tf.equal(tf.argmax(y,axis=1),tf.argmax(y_data,axis=1))
#            acc = sess.run(pre,feed_dict={x_data:data_te,y_data:labels_tes})
#            print(i,'测试: ',sum(acc)/len(acc))
#        for j in np.random.randint(0,4799,size=[100]):
#            x_s[k] = data[j,:]
#            y_s[k] = labels_trai[j]
#            k = k+1
#        sess.run(train,feed_dict={x_data: x_s, y_data: y_s})
#