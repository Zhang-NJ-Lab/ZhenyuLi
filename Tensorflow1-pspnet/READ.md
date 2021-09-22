1.配置
Keras                2.1.5
tensorflow-gpu       1.13
python               3.7.9
CUDA Toolkit         10.0
labelme              3.16 
numpy                1.19.5

2.操作流程
（1）将labelme标注的样本放入dataset_processing中的before，修改json_to_dataset中的种类，运行json_to_dataset.py
（2）修改dataset_classfication中的训练集比例，运行dataset_classfication.py
（3）修改train中的图片格式和分类个数，运行train.py
（4）修改pspnet中的模型地址和标注参数
（5）运行predict.py，输入文件地址