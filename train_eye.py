# 导入相关库
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras import backend as K
import numpy as np
import myModel

def parse_voc_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    a = 0
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        a+=1
        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
        if len(img['object']) > 0:
            all_imgs += [img]
    print(a)
    return all_imgs, seen_labels

def parse_filename(filename):
    # Split the filename
    parts = filename.split('_')
    
    # Get the eye state label
    eye_state = int(parts[4])
    
    # Return the label
    return eye_state

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
learning_rate = 0.0005
input_shape = (32, 32, 1)  # 赋值为你的输入形状

# 标签
# labels = ["closed_eye", "closed_mouth", "open_eye", "open_mouth"]
# img_dir = 'C:/Users/dwc20/OneDrive/Desktop/dataset/JPEGImages/'
# ann_dir = 'C:/Users/dwc20/OneDrive/Desktop/dataset/Annotations/'

# 解析注释文件
# all_imgs, seen_labels = parse_voc_annotation(ann_dir, img_dir, labels)

# 初始化训练数据和标签列表
data = []
labels = []

# 循环遍历所有的图像
# for img in all_imgs:
#     # 加载图像并预处理
#     image = load_img(img['filename'], target_size=input_shape, color_mode='grayscale')
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=-1)

#     # 更新数据和标签列表
#     data.append(image)
#     labels.append([obj['name'] for obj in img['object']])  # 添加所有标签

# 将数据和标签转为numpy数组
# data = np.array(data, dtype="float32") / 255.0
# labels = np.array(labels, dtype=object)

# 对标签进行编码
# mlb = MultiLabelBinarizer()
# labels = mlb.fit_transform(labels)

# Define the dataset directory
dataset_dir = 'C:/Users/dwc20/OneDrive/Desktop/mrlEyes_2018_01/'

# Loop over the dataset directory
for subdir, dirs, files in os.walk(dataset_dir):
    for filename in files:
        # Get the full path of the current file
        filepath = subdir + os.sep + filename
        
        # Load the image
        image = load_img(filepath, target_size=input_shape, color_mode='grayscale')
        image = img_to_array(image)
    
        # Get the label from the filename
        label = parse_filename(filename)

        # Add the data and labels to the lists
        data.append(image)
        labels.append(label)
        
        print(filename)

# Convert the data and labels to numpy arrays
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Perform one-hot encoding on the labels
labels = to_categorical(labels, num_classes=2)


# 划分训练集和测试集
print("split start")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print("split done")

# Set up your list of batch sizes
batch_sizes = [16, 32, 64]

# Loop over your batch sizes
for batch_size in batch_sizes:

    # 创建TensorBoard回调对象，设置日志目录
    log_dir = os.path.join('C:/Users/dwc20/OneDrive/Desktop/face_detect/TrainLog/', f'batch_size_{batch_size}')
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
    
    model.save(f'batch_size_{batch_size}.h5')