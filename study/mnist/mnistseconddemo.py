#mnist进阶例子
#导入TensorFlow和input
import tensorflow as tf
import  study.mnist.input_data as inputdata

#声明占位符，shape可以不指定，但是指定后Python会自动检查维度不匹配的情况
x = tf.placeholder("float",shape=[None,784]) #用于输入变量
y_ = tf.placeholder("float",shape=[None,10]) #用于输入标签

#声明运算参数变量，程序目标就是计算出该变量的合适取值
w = tf.Variable(tf.zeros([784,10]))#这里的初始值可以是任意值，我们都默认为0
b = tf.Variable(tf.zeros([10]))

#我们创建一个InteractionSession，该会话可以在计算中修改图形
sess = tf.InteractiveSession()
#变量需要在session中初始化
sess.run(tf.initialize_all_variables())
#构建类别预测和损失函数，我们用softmax作为分类算法
y = tf.nn.softmax(tf.matmul(x,w) + b)
#通过交叉熵评估损失函数
cross_entry = -tf.reduce_sum(y_*tf.log(y))
#训练模型，我们用最速下降法，步长为0.01，算法为GradientDescentOptimizer来逼近取值
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entry)#目标是cross_entry的最小值
#开始训练，首先读取数据
mnist = inputdata.read_data_sets("F:/pythonworkspace/mnistdata",one_hot=True)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train.run(feed_dict={x:batch[0],y_:batch[1]})
#评估模型，使用预测概率计算
correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#求取正确的平均值
res = tf.reduce_mean(tf.cast(correct,"float"))
print(sess.run(res,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))