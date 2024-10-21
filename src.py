# import pandas as pd

# df = pd.read_csv("/content/all_letters_info.csv")
# print(df[{'letter','label'}].head())

# nullcoun=df.isnull().sum
# print(nullcoun)

# df.dtypes

# df.info()

! pip install -q kaggle

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

!sudo mkdir -p /root/.kaggle/ && sudo mv kaggle.json /root/.kaggle/ && sudo chmod 600 /root/.kaggle/kaggle.json


! kaggle datasets download -d "tatianasnwrt/russian-handwritten-letters"

!unzip /content/russian-handwritten-letters.zip

import os
# i=0
for dirname, _, filenames in os.walk('/content/all_letters_image'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        # i=i+1


# print(i)

# load the libraries
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import os
import h5py
import PIL
import cv2
import tensorflow as tf
import tensorflow.keras as keras

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pylab as plt
from matplotlib import cm
%matplotlib inline

input_folder = '/content/all_letters_image/all_letters_image/'
all_letters_filename = os.listdir(input_folder)
len(all_letters_filename)

i = Image.open("/content/all_letters_image/all_letters_image/01_01.png")
i

i_arr = np.array(i)
i_arr

# Helper functions to preprocess an image into a tensor.
# We will use the default RGB mode
#instead of a possible RGBA as the opacity doesn't seem to be important in this task (still should be tested)

#TO DO: describe the function

def img_to_array(img_name, input_folder):
    img = image.load_img(input_folder + img_name, target_size=(32,32))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)
def data_to_tensor(img_names, input_folder):
    list_of_tensors = [img_to_array(img_name, input_folder) for img_name in img_names]
    return np.vstack(list_of_tensors)

data = pd.read_csv("/content/all_letters_info.csv")
image_names = data['file']
letters = data['letter']
backgrounds = data['background'].values
targets = data['label'].values
tensors = data_to_tensor(image_names, input_folder)

tensors[0]

# Print the shape
print ('Tensor shape:', tensors.shape)
print ('Target shape', targets.shape)

# Read from files and display images using OpenCV
def display_images(img_path, ax):
    img = cv2.imread(input_folder + img_path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

fig = plt.figure(figsize=(16, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i + 1, xticks=[], yticks=[], title=letters[i*100])
    display_images(image_names[i*100], ax)

X = tensors.astype("float32")/255


arr = X[0]
arr_ = np.squeeze(arr)
plt.imshow(arr_)
plt.show()

g = plt.imshow(X[0][:,:,0])

img_rows, img_cols = 32, 32 # because our pictures are 32 by 32 pixels
num_classes = 33 # because there are 33 letters in the Russina alphabet
y = keras.utils.to_categorical(targets-1, num_classes)

print(X.shape)
print(y.shape)

def captch_ex(file_name):
    img = cv2.imread(file_name)
    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV

    '''
            line  8 to 12  : Remove noisy portion
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    # for cv2.x.x

    contours, hierarchy = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.png'
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    cv2.imshow('captcha_result', img)
    cv2.waitKey()

file_name = '/kaggle/input/russian-handwritten-letters/all_letters_image/all_letters_image/04_100.png'
# captch_ex(file_name)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.15, # Randomly zoom image
    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Split the data into train, validation and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

y_val=np.argmax(y_val, axis=1)


# Define the model architecture

deep_RU_model = Sequential()

deep_RU_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (img_rows,img_cols,3)))
deep_RU_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
deep_RU_model.add(MaxPooling2D(pool_size=(2,2)))
deep_RU_model.add(Dropout(0.25))


deep_RU_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
deep_RU_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
deep_RU_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
deep_RU_model.add(Dropout(0.25))


deep_RU_model.add(Flatten())
deep_RU_model.add(Dense(256, activation = "relu"))
deep_RU_model.add(Dropout(0.5))
deep_RU_model.add(Dense(33, activation = "softmax"))

# Define the optimizer
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
from tensorflow.keras.optimizers import legacy

optimizer = legacy.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# Compile the model:

deep_RU_model.compile(loss="categorical_crossentropy", optimizer = optimizer,metrics=["accuracy"])

history = deep_RU_model.fit(X_train, y_train, batch_size=90,epochs=24)

print(history.history.keys())


# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
# ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
# ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# Plot the Neural network fitting history
def history_plot(fit_history, n):
    plt.figure(figsize=(18, 12))

    plt.subplot(211)
    plt.plot(fit_history.history['loss'][n:], color='slategray', label = 'train')
#     plt.plot(fit_history.history['val_loss'][n:], color='#4876ff', label = 'valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Loss Function');

    plt.subplot(212)
    plt.plot(fit_history.history['accuracy'][n:], color='slategray', label = 'train')
#     plt.plot(fit_history.history['val_categorical_accuracy'][n:], color='#4876ff', label = 'valid')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title('Accuracy');


history_plot(history, 0)

y_pred = deep_RU_model.predict(X_val)
y_pred=np.argmax(y_pred, axis=1)

val_acc_score_1 = round(accuracy_score(y_val,y_pred) * 100, 2)
val_acc_score_1

# Load and preprocess the image
img_path = '/content/300px-Russian-Alphabet-Letters-Polyglotclub.jpg'
img = image.load_img(img_path, target_size=(32, 32))  # Assuming input size is 32x32
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = img_batch / 255.0  # Normalize pixel valu

predictions = deep_RU_model.predict(img_preprocessed)

# Print the predicted class with highest probability
print(predictions.argmax())
# Get the predicted class index and its probability
predicted_class_index = predictions.argmax()
predicted_class_probability = predictions[0][predicted_class_index] * 100  # Multiply by 100 for percentage

# Print the predicted class and its percentage
print("Predicted class:", predicted_class_index)
print("Prediction percentage:", predicted_class_probability, "%")

