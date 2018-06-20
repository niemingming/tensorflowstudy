#TensorFlow例子练习
import study.mnist.input_data as inputdata

datas = inputdata.read_data_sets("F:/pythonworkspace/mnistdata",one_hot=True)

print(datas)