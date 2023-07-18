# 导入相关库
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras import backend as K
import numpy as np
import myModel

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# 设置一些基础参数
batch_size = 32
num_classes = 2  # 赋值为你的类别数量
epochs = 100
learning_rate = 0.0001
input_shape = (32, 32, 1)  # 赋值为你的输入形状



# 初始化训练数据和标签列表
data = []
labels = []


# Define the dataset directory
dataset_dir = 'C:/Users/dwc20/OneDrive/Desktop/mouth/'

# Loop over the dataset directory
for subdir, dirs, files in os.walk(dataset_dir):
    # Get the label from the directory name
    label = os.path.basename(subdir)
    for filename in files:
        # Get the full path of the current file
        filepath = subdir + os.sep + filename

        # Load the image
        image = load_img(filepath, target_size=input_shape, color_mode='grayscale')
        image = img_to_array(image)
        
        # Add the data and labels to the lists
        data.append(image)
        labels.append(label)
        
        print(filename)

# Convert the labels to binary format
labels = np.array([1 if label == 'open' else 0 for label in labels])

# Convert the data and labels to numpy arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Perform one-hot encoding on the labels
labels = to_categorical(labels, num_classes=2)


# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Set up your list of batch sizes
learning_rates = [0.0005]


# 创建TensorBoard回调对象，设置日志目录
log_dir = os.path.join('C:/Users/dwc20/OneDrive/Desktop/face_detect/TrainLog/', f'test_mouth')
tensorboard_callback = TensorBoard(log_dir=log_dir)

# 建立模型
base_model = myModel.CNN.build(input_shape[0], input_shape[1], input_shape[2], num_classes)
# base_model = load_model("C:/Users/dwc20/best0428ep150.h5")

# 移除模型的最后一层（假设是输出层）
base_model.layers.pop()

# 添加一个新的全连接层作为输出层，该层的神经元数量与目标类别数量一致
x = base_model.layers[-1].output
predictions = Dense(num_classes, activation='softmax')(x)

# 创建新模型
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型，设置损失函数、优化器和评估指标
model.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
            metrics=['accuracy', f1_score])


# 自定义回调函数用于输出训练进程
class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

# 创建自定义回调对象
progress_callback = TrainingProgressCallback()

# 训练模型
print("start training")
model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[progress_callback, tensorboard_callback])
print("training finished")

# 在测试数据上评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(f'test_mouth.h5')