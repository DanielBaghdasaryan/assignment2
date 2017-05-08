import numpy as np

def vectorize(a):  #E.g. input: 4; output: [0,0,0,0,1,0,0,0,0,0]
	a_vectors=[]
	for i in range(len(a)):
		a_vector=np.zeros(10)
		a_vector[int(a[i])]=1
		a_vectors.append(a_vector)
	a_vectors=np.array(a_vectors)
	return a_vectors


def test(network,X,Y):
	errors = 0
	i=0
	print("true\tpredict")
	for img, true_label in zip(X, Y):
		i+=1
		out_v = network.forward_propagate(img)
		errors += 0 if np.argmax(out_v) == np.argmax(true_label) else 1
		if i<20:
			print(str(np.argmax(out_v))+"\t"+str(np.argmax(true_label)))
	print('Error: {}%'.format(100. * errors / i))

