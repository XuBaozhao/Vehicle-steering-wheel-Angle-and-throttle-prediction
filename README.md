# Vehicle-steering-wheel-Angle-and-throttle-prediction
Vehicle steering wheel Angle and throttle prediction
数据来源：

数据来源是unity公司的模拟器，通过自己驾驶录制，会产生图片文件和csv文件
csv文件包括：center、left、right、steering、throttle、brake、speed
我们只需要使用其中的center、left、right三列即可。
#需求分析

通过给定图片和csv文件信息，训练一个model，该model能够通过输入一张图片预测方向盘转角和油门大小。

#设计 

读取csv文件信息，判断csv文件行 == 图片数量 * 3
将需要的三列数据提取出来，并shuffle，通过train_test_split分割训练集和验证集
设置输入图片大小，根据输入大小构建CNN模型
ModelCheckpoint保存最优模型、EarlyStopping检测变量、fit_generator训练模型
保存model
通过model进行test
