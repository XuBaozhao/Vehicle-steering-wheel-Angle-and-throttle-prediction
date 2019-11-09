# 科学计算
import numpy as np
# 优化算法
from keras.optimizers import Adam
# 全连接层
from keras.layers.core import Dense
# 卷积、池化、展平
from keras.layers import Conv2D, MaxPooling2D, Flatten
# 激活函数
from keras.layers.advanced_activations import LeakyReLU
# 输入、模型
from keras import Input, Model
# 文件处理
import csv
# 图片处理
import cv2
# 查看文件下文件信息，通配符查找
import glob
# 数据存储
import pickle
# 洗牌
from sklearn.utils import shuffle
# 数据机切分
from sklearn.model_selection import train_test_split
# 回调函数
from keras import callbacks

SEED = 13

def horizontal_flip(img, degree):
    '''
    按照50%的概率水平翻转图像
    img: 输入图像
    degree: 输入图像对于的转动角度
    '''
    choice = np.random.choice([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return (img, degree)


def random_brightness(img, degree):
    '''
    随机调整输入图像的亮度， 调整强度于 0.1(变黑)和1(无变化)之间
    img: 输入图像
    degree: 输入图像对于的转动角度
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 调整亮度V: alpha * V
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)
    v = hsv[:, :, 2]
    v = v * alpha
    hsv[:, :, 2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)

    return (rgb, degree)


def left_right_random_swap(img_address, degree, degree_corr=1.0 / 4):
    '''
    随机从左，中，右图像中选择一张图像， 并相应调整转动的角度
    img_address: 中间图像的的文件路径
    degree: 中间图像对于的方向盘转动角度
    degree_corr: 方向盘转动角度调整的值
    '''
    swap = np.random.choice(['L', 'R', 'C'])

    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        corrected_label = np.arctan(np.tan(degree) + degree_corr)
        return (img_address, corrected_label)
    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        corrected_label = np.arctan(np.tan(degree) - degree_corr)
        return (img_address, corrected_label)
    else:
        return (img_address, degree)


def discard_zero_steering(degrees, rate):
    '''
    从角度为零的index中随机选择部分index返回
    degrees: 输入的角度值向量
    rate: 丢弃率， 如果rate=0.8， 意味着 80% 的index会被返回，用于丢弃
    '''
    # degrees为0的所有地址
    steering_zero_idx = np.where(degrees == 0)
    steering_zero_idx = steering_zero_idx[0]
    # 删除个数 = 为0的degree个数 * 丢弃率
    size_del = int(len(steering_zero_idx) * rate)
    # 返回选择出的degree为0的
    return np.random.choice(steering_zero_idx, size=size_del, replace=False)


def get_model(shape):
    '''
    预测方向盘角度: 以图像为输入, 预测方向盘的转动角度和油门throttle
    shape: 输入图像的尺寸, 例如(128, 128, 3)
    '''
    input_1 = Input(shape, name='input_1')

    conv_1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu')(input_1)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    conv_2 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_1)
    maxpool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_2)
    maxpool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    conv_4 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_3)
    maxpool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_4)
    maxpool_5 = MaxPooling2D(pool_size=(2, 2))(conv_5)

    conv_6 = Conv2D(512, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_5)
    maxpool_6 = MaxPooling2D(pool_size=(2, 2))(conv_6)

    conv_7 = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", activation='relu')(maxpool_6)
    maxpool_7 = MaxPooling2D(pool_size=(2, 2))(conv_7)

    flat = Flatten()(maxpool_7)

    dense_1 = Dense(256, activation='relu')(flat)
    dense_2 = Dense(512, activation='relu')(dense_1)
    dense_3 = Dense(10, activation='relu')(dense_2)

    output_1 = Dense(1, activation='linear', name='output_1')(dense_3)
    output_2 = Dense(1, name='output_2')(dense_3)
    output_2_ = LeakyReLU(alpha=0.05)(output_2)

    model = Model(inputs=[input_1], outputs=[output_1, output_2_])
    sgd = Adam(lr=0.0001)
    model.compile(optimizer=sgd, loss=['mean_squared_error', 'mean_squared_error'],
                  loss_weights=[1.0, 0.8])
    return model


# 图像数据增强
def image_transformation(img_address, degree, data_dir):
    img_address, degree = left_right_random_swap(img_address, degree)  # 图像的左右翻转
    img = cv2.imread(data_dir + img_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, degree = random_brightness(img, degree)  # 图像亮度调整
    img, degree = horizontal_flip(img, degree)  # 图像水平翻转

    return (img, degree)

# 生成器
def generator_data(img, degree, throttle, batch_size, shape, data_dir='', discard_rate=0.65):
    '''
    :param img: X_train
    :param degree: y_train
    :param throttle: z_train
    :param batch_size: batch_size
    :param shape: shape
    '''
    y_bag = []
    z_bag = []
    # 洗牌
    x, y, z = shuffle(img, degree, throttle)
    # 丢弃65%的角度为0的数据
    rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
    # 删除对应id的行
    new_x = np.delete(x, rand_zero_idx, axis=0)
    new_y = np.delete(y, rand_zero_idx, axis=0)
    new_z = np.delete(z, rand_zero_idx, axis=0)

    offset = 0
    # 死循环
    while True:
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))
        Z = np.empty((batch_size, 1))

        for example in range(batch_size):
            # 对每个图片的信息
            img_address, img_steering, img_throttle = new_x[example + offset], new_y[example + offset], new_z[
                example + offset]
            # 图片数据增强
            img, img_steering = image_transformation(img_address, img_steering, data_dir)
            # 修改图片大小
            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5

            Y[example] = img_steering
            Z[example] = img_throttle
            y_bag.append(img_steering)
            z_bag.append(img_throttle)
            '''
             到达原来数据的结尾, 从头开始
            '''
            if (example + 1) + offset > len(new_y) - 1:
                x, y, z = shuffle(x, y, z)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_z = z
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                new_z = np.delete(new_z, rand_zero_idx, axis=0)
                offset = 0
        yield (X, [Y, Z])

        offset = offset + batch_size

        np.save('y_bag.npy', np.array(y_bag))
        np.save('z_bag.npy', np.array(z_bag))
        np.save('Xbatch_sample.npy', X)

'''
数据来源是unity公司的模拟器，通过自己驾驶录制，会产生图片文件和csv文件
csv文件包括：center、left、right、steering、throttle、brake、speed
'''
if __name__ == "__main__":
    # cvs文件路径
    data_path = '/Users/qishi/Documents/car/'
    # 打开csv文件
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        # 读文件
        file_reader = csv.reader(csvfile, delimiter=',')
        # 把文件信息放在log列表中
        log = []
        for row in file_reader:
            log.append(row)
    # 将列表转化为数组
    log = np.array(log)
    # 去掉文件第一行
    # log = log[0:, :]

    # 判断图像文件数量是否等于csv日志文件中记录的数量
    ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    assert len(ls_imgs) == len(log) * 3, 'number of images does not match'

    # 使用20%的数据作为测试数据
    validation_ratio = 0.2
    # 图片大小
    shape = (128, 128, 3)
    # 每训练一次训练几张图片
    batch_size = 32
    # 训练次数，之后改为32
    nb_epoch = 32

    # 图像文件名
    x_ = log[:, 0]
    # 转角
    y_ = log[:, 3].astype(float)
    # 油量
    z_ = log[:, 4].astype(float)
    print(x_.shape)
    print(y_.shape)
    print(z_.shape)
    # 一起洗牌，对应信息不变
    x_, y_, z_ = shuffle(x_, y_, z_)
    # 训练集和验证集，20%验证，80%测试
    # random_state就是为了保证程序每次运行都分割一样的训练集和测试集。否则，同样的算法模型在不同的训练集和测试集上的效果不一样
    # 当你用sklearn分割完测试集和训练集，确定模型和初始参数以后，你会发现程序每运行一次，都会得到不同的准确率，无法调参。这个时候就是因为没有加random_state。加上以后就可以调参了
    X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(x_, y_, z_, test_size=validation_ratio,
                                                                      random_state=SEED)

    print('batch size: {}'.format(batch_size))
    print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))

    # 每次训练的样本数 = 批次
    samples_per_epoch = batch_size
    # 使得validation数据量大小为batch_size的整数倍
    nb_val_samples = len(y_val) - len(y_val) % batch_size
    # 获得训练model
    model = get_model(shape)
    # 查看模型状态
    print(model.summary())

    # 根据validation loss保存最优模型
    save_best = callbacks.ModelCheckpoint('best_model_self.h5', # 模型位置和名称
                                          monitor='val_loss', # 想要监视的值 val_acc / val_loss / acc / loss
                                          verbose=1, # 想要进度条选1，否则选0
                                          save_best_only=True, # 只保存最好的模型
                                          mode='min') # val_loss:min / val_acc:max / 一般情况下选auto自动推断

    # 如果训练持续没有validation loss的提升, 提前结束训练 / 当监测值不再改善时，该回调函数将中止训练
    early_stop = callbacks.EarlyStopping(monitor='val_loss', # 监视的量
                                         min_delta=0, # 增大或减小的阈值，只有大于这个部分才算作improvement
                                         patience=50, # 如果没有改善、则经过patience后停止训练 / 能够容忍多少个epoch内都没有improvement
                                         verbose=0, # 进度条
                                         mode='min') # 同上
    callbacks_list = [early_stop, save_best]

    # 数据集小时用fit，因为可以全部放进内存，然后不用数据增强
    # 真实世界的数据集通常太大而无法放入内存中 / 它们也往往具有挑战性，要求我们执行数据增强以避免过拟合并增加我们的模型的泛化能力
    # 采用fit_generator函数
    history = model.fit_generator(generator_data(X_train, y_train, z_train, batch_size, shape), # 生成器，一个yield的函数，迭代返回数据
                                  steps_per_epoch=samples_per_epoch, # 一次训练周期里面进行多少次batch
                                  # 两个反斜杠会直接取整数
                                  validation_steps=nb_val_samples // batch_size, # 设置验证多少次数据后取平均值作为此epoch训练后的效果，val_loss,val_acc的值受这个参数直接影响
                                  validation_data=generator_data(X_val, y_val, z_val, batch_size, shape), # 验证用的数据源设置，evaluate_generator函数要用到这个数据源，可以使用生成器
                                  epochs=nb_epoch, # 训练几轮
                                  verbose=1, # 一个开关，打开时，打印清晰的训练数据，即加载ProgbarLogger这个回调函数
                                  callbacks=callbacks_list) # 设置业务需要的回调函数，我们的模型中添加了ModelCheckpoint这个回调函数

    with open('./trainHistoryDict_self.p', 'wb') as file_pi:
        # 把history的history信息存入上面文件
        # 记录了val_loss信息
        pickle.dump(history.history, file_pi)

    # 通过json文件保存model
    with open('model_self.json', 'w') as f:
        f.write(model.to_json())

    # 同时保存model和权重的方式
    model.save('model_self.h5')

    # img = cv2.imread('test_pic.jpg')
    # img = cv2.resize(img, (128, 128))
    # X = np.empty((1, *shape))
    # X[0,:,:,:] = cv2.resize(img,(128,128)) / 255 - 0.5
    #
    # prediction_data = model.predict(X)
    # print('prediction_data: ',prediction_data)
    print('Done!')

