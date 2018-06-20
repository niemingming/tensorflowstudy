#多层卷积网络，用于优化mnist获取更好的预测。
import tensorflow as tf
import study.mnist.input_data as inputdata
#定义函数
#该函数表示生成标准差为0.1，维度为shape的张量
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape,stddev=0.1)#truncated_normal是生成张量函数，其中shape是维度，mean是平均值，stddev是标准差
    return tf.Variable(initial)
#定义偏移量生成函数
#表示生成维度为shape的张量常量，量值为0.1，我们使用的是ReLUctant神经元，因此激活函数需要一个较小的正数偏移量，防止0梯度出现
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#定义卷积函数，我们定义步长stride为1，padding边距为0的模板，保证输入输出的维度一致,我们是黑白的，因此是一维的
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")
#我们采用传动的2X2模板，做max pooling，ksize定义模板大小，后三位是维度，stride是步长
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1],padding="SAME")
#定义占位符和变量
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])
#定义变量
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#创建会话
sess = tf.InteractiveSession()
#读取数据
mnist = inputdata.read_data_sets("F:/pythonworkspace/mnistdata",one_hot=True)
#第一层卷积
w_conv1 = weight_variable([5,5,1,32])#第一层卷积，采用5x5的视图（过滤）矩阵，1通道，生成32个特征数据
b_conv1 = bias_variable([32])
#需要将x数据转成2维数据，因为是灰色图片，颜色通道为1，
x_image = tf.reshape(x,[-1,28,28,1])#我们将图片还原为28x28的图片，
#我们将x_image和w进行卷积，然后通过ReLUctant激活函数，最后进行2x2的最大池化
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第二层卷积，因为我们第一层生成了32个特征值
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#密集连接层，该层输出最终的分类
#经过两轮卷积池化形成了7x7的矩阵，64个特征，我们引入1024个神经元的全连接
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
#将结果reshape标准格式
h_pool_fc = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_fc,w_fc1) + b_fc1)
#为了减少拟合过程，我们在输出层前加一个dropout，我们用placeholder来代表神经元输出在dropout中保持不变的概率
#tf的dropout函数除了计算中的dropout之外，还会自动考虑scale，因此使用该函数不需要考虑scale
keep_pro = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_pro)
#输出层，我们通过该层做分类
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)

#训练和评估
#我们使用ADAM优化器进行优化
cross_entry = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entry)
#评估函数
correct = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
#评估结果
accroy = tf.reduce_mean(tf.cast(correct,"float"))
#运行计算
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        res = sess.run(accroy,feed_dict={x:batch[0],y_:batch[1],keep_pro:1.0})
        print("step %d, training accuracy %g" % (i, res))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_pro:0.5})

testbatch = mnist.test.next_batch(200)
res = sess.run(accroy,feed_dict={x:testbatch[0],y_:testbatch[1],keep_pro:1.0})
print("test, training accuracy %g" % ( res))

