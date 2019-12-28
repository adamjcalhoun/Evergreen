from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_name = '2019-12-24-14-03-45.h5'

model = load_model(model_name)

whitenoise = np.random.random((10000,10))

prediction = model.predict(whitenoise)

# print(prediction.shape)
# print(np.argmax(prediction,axis=1))

plt.plot(np.mean(whitenoise[np.argmax(prediction,axis=1)==0],axis=0),'b')
plt.plot(np.mean(whitenoise[np.argmax(prediction,axis=1)==1],axis=0),'r')
plt.plot(np.mean(whitenoise[np.argmax(prediction,axis=1)==2],axis=0),'g')
plt.plot(np.mean(whitenoise[np.argmax(prediction,axis=1)==3],axis=0),'k')

plt.show()

