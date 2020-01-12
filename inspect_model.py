import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_name = 'memory_2020-01-04-21-19-23.pkl'
model_name = 'memory_2020-01-04-21-34-26.pkl'
####
# with h5py.File(model_name,'r') as f:
# 	print(f.keys())
# 	print(f['memory'])

action = []
C = []
dCdt = []
with open(model_name, 'rb') as handle:
    b = pickle.load(handle)

    for ii in range(len(b)):
    	C.append(b[ii][0][0])
    	dCdt.append(b[ii][0][0] - b[ii][0][1])
    	action.append(b[ii][1])

# print(C)
# print(dCdt)
# print(action)

C = np.array(C)
dCdt = np.array(dCdt)
action = np.array(action)

# print((np.mean(action[C < 0.1] == 0),np.mean(action[C > 0.1] == 0)))
# print((np.mean(action[C < 0.1] == 1),np.mean(action[C > 0.1] == 1)))
# print((np.mean(action[C < 0.1] == 2),np.mean(action[C > 0.1] == 2)))
# print((np.mean(action[C < 0.1] == 3),np.mean(action[C > 0.1] == 3)))

hbs = np.arange(-0.5,0.6,0.2)
# print(hbs)
data = np.zeros((hbs.shape[0],4))
for ind,edges in enumerate(hbs):
	for aa in range(4):
		data[ind,aa] = np.mean(action[np.logical_and(dCdt > edges,dCdt <= edges+0.1)] == aa)/np.mean(action[np.logical_and(dCdt > edges,dCdt <= edges+0.1)])
plt.plot(hbs,data[:,2] + data[:,3])
plt.show()
exit()

hist,bin_edges = np.histogram(dCdt,bins=10)
plt.plot(bin_edges[1:],hist)
plt.show()

exit()

hbs = np.arange(0.01,1,0.1)
data = np.zeros((hbs.shape[0],4))
for ind,edges in enumerate(hbs):
	for aa in range(4):
		data[ind,aa] = np.mean(action[np.logical_and(C > edges,C <= edges+0.1)] == aa)/np.mean(action[np.logical_and(C > edges,C <= edges+0.1)])

# plt.plot(data[:,0]+data[:,1])
plt.plot(data[:,2] + data[:,3])


# plt.plot(data[:,0])
# plt.plot(data[:,1])
# plt.plot(data[:,2])
# plt.plot(data[:,3])

# hist,bin_edges = np.histogram(C[C > 0],bins=10)
# plt.plot(bin_edges[1:],hist)
plt.show()
exit()


######
# from tensorflow.keras.models import load_model
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


