#引入TensorFlow
import tensorflow as tf
import study.mnist.input_data as inputdata

#定义符号变量
x = tf.placeholder("float",[None,784])
#这里x不是一个特定的值，而是占位符，placeholder，在计算时输入这个值，我们用二位张量表示这些图，其中784表示了图是784维的，None表示张量第一维度是任意的，也就是可以是任意个图
#除了上面的占位符，我们还可以使用Variable创建一个可修改的张量
w = tf.Variable(tf.zeros([784,10]))#我们用全是0的张量来初始化变量
b = tf.Variable(tf.zeros([10]))
#通过tf的模型实现softmax模型
y = tf.nn.softmax(tf.matmul(x,w) + b)#matmul表示x与w的乘积，x是784维的向量
#训练模型
#用交叉熵做评估，交叉熵需要进一步了解,交叉熵用于效果评估，其中y_表示真实概率，y表示计算概率
#定义占位符
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#GradientDescentOptimizer是学习速率。我们这里指定按照0.0.1的速率进行梯度学习
#初始化变量
init = tf.initialize_all_variables()
#我们在session里面启动模型，并初始化变量
sess = tf.Session()
sess.run(init)
#读取模型
datas = inputdata.read_data_sets("F:/pythonworkspace/mnistdata",one_hot=True)
#训练模型
for i in range(1000):
    batch_xs,batch_ys = datas.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
#评估
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print (sess.run(accuracy,feed_dict={x:datas.test.images,y_:datas.test.labels}))
