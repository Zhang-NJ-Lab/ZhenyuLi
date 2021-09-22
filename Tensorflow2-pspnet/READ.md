1.这是一个基于tensorflow2的PSPnet模型

2.配置
tensorflow-gpu       2.2
python               3.7.9
CUDA Toolkit         10.1
cudnn                7.6.5.32
labelme              3.16 

3.操作流程
（1）将labelme标注的样本放入dataset_processing中的before，修改json_to_dataset中的种类，运行json_to_dataset.py
（2）修改dataset_classfication中的训练集比例，运行dataset_classfication.py
（3）修改train中的图片格式和分类个数，运行train.py
（4）修改pspnet中的模型地址和标注参数
（5）运行predict.py，输入文件地址