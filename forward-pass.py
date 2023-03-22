import math

def relu(x):
	return max(0,x)

def sigmoid(x):
	if x <= -700:
		return 0
	if 700 > x > -700:
		return 1/(1+(math.exp(-x)))
	if x >= 700:
		return 1

def forward_pass(Network,X):
	for lay in Network:
		L = []
		if lay[0].find("linear") != -1:
			for w in lay[1]:
				sum=0
				i = 0
				while i < len(X):
					sum=sum+(X[i]*w[i])
					i+=1
				L.append(sum)
			X = L
		
		elif lay.find("relu") != -1:
			for a in X:
				L.append(relu(a))
			X = L

		elif lay.find("sigmoid") != -1:
			for a in X:
				L.append(sigmoid(a))
			X = L
	return X
