from keras.models import load_model
import cv2
import numpy as np

model = load_model('best_model_self.h5')
img = cv2.imread('left_2019_11_08_18_28_36_304.jpg')
shape = (128, 128, 3)
img = cv2.resize(img, (128, 128))
X = np.empty((1, *shape))
X[0,:,:,:] = cv2.resize(img,(128,128)) / 255 - 0.5

prediction_data = model.predict(X)
print('prediction_data: ',prediction_data)