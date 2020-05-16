import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_name = 'memory_2020-01-04-21-19-23.pkl'
model_name = 'memory_2020-01-04-21-34-26.pkl'
model_name = 'memory_2020-01-27-13-18-30.pkl'
####
# with h5py.File(model_name,'r') as f:
# 	print(f.keys())
# 	print(f['memory'])

action = []
C = []
dCdt = []
with open(model_name, 'rb') as handle:
    b = pickle.load(handle)

    # print(b)
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

# # print((np.mean(action[C < 0.1] == 0),np.mean(action[C > 0.1] == 0)))
# # print((np.mean(action[C < 0.1] == 1),np.mean(action[C > 0.1] == 1)))
# # print((np.mean(action[C < 0.1] == 2),np.mean(action[C > 0.1] == 2)))
# # print((np.mean(action[C < 0.1] == 3),np.mean(action[C > 0.1] == 3)))

# # print((np.max(dCdt),np.min(dCdt)))
# # print(dCdt)
# # hbs = np.arange(-0.5,0.6,0.2)
# hbs = np.arange(-0.01,0.01,0.001)
# # print(hbs)
# data = np.zeros((hbs.shape[0],4))
# for ind,edges in enumerate(hbs):
# 	for aa in range(4):
# 		data[ind,aa] = np.mean(action[np.logical_and(dCdt > edges,dCdt <= edges+0.1)] == aa)/np.mean(action[np.logical_and(dCdt > edges,dCdt <= edges+0.1)])

# plt.plot(hbs,data[:,2] + data[:,3])
# plt.show()
# exit()

# hist,bin_edges = np.histogram(dCdt,bins=10)
# plt.plot(bin_edges[1:],hist)
# plt.show()

# exit()

print((np.max(C),np.min(C)))
hbs = np.arange(0.0,0.01,0.001)
data = np.zeros((hbs.shape[0],4))
for ind,edges in enumerate(hbs):
	for aa in range(4):
		data[ind,aa] = np.mean(action[np.logical_and(C > edges,C <= edges+0.1)] == aa)/np.mean(action[np.logical_and(C > edges,C <= edges+0.1)])

# plt.plot(data[:,0]+data[:,1])
plt.plot(hbs,data[:,2] + data[:,3])


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
# # model_name = 'twopatch_diffuse_rfh_5.h5'
# model_name = 'twopatch_rfh_5.h5'
# state_space = 31
# model = load_model(model_name)

# whitenoise = (np.random.random((10000,state_space))-0.5)*10

# prediction = model.predict(whitenoise)
# # print(prediction)

# # print(prediction.shape)
# # print(np.argmax(prediction,axis=1))
# # ACTION = ['forward','backward','left','right']

# # plt.plot(np.mean(whitenoise[np.argmax(prediction,axis=1)==0],axis=0),'b')
# plt.plot(np.mean(whitenoise[np.argmax(prediction,axis=1)==1],axis=0),'r')
# plt.plot(np.mean(whitenoise[np.argmax(prediction,axis=1)==2],axis=0),'g')
# # plt.plot(np.mean(whitenoise[np.argmax(prediction,axis=1)==3],axis=0),'k')

# plt.show()


