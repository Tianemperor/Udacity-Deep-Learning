#coding:utf-8
import numpy as np
class Neuron:
	def __init__(self, inbound_neurons=[]):
		self.inbound_neurons = inbound_neurons
		self.outbound_neurons = []
		self.value = None
		self.gradients = {}
		for n in self.inbound_neurons:
			n.outbound_neurons.append(self)

	def forward(self):
		return NotImplemented

	def backward(self):
		return NotImplemented

class Input(Neuron):
	"""docstring for Input"""
	def __init__(self):
		"""
		Input neuron不需要从其他节点接收输入
		"""
		Neuron.__init__(self)

	def forward(self, value=None):
		if value is not None:
			self.value = value

	def backward(self):
		self.gradients = {self: 0}
		for n in self.outbound_neurons:
			self.gradients[self] += n.gradients[self]

class Add(Neuron):
	def __init__(self, *inputs):
		Neuron.__init__(self, inputs)

	def forward(self):
		x_value = self.inbound_neurons[0].value
		y_value = self.inbound_neurons[1].value
		self.value = x_value + y_value

class Mult(Neuron):
	def __init__(self, *inputs):
		Neuron.__init__(self, inputs)

	def forward(self):
		x_value = self.inbound_neurons[0].value
		y_value = self.inbound_neurons[1].value
		self.value = x_value * y_value

class Linear(Neuron):
	def __init__(self, X, W, b):
		Neuron.__init__(self, [X, W, b])

	def forward(self):
		#inputs = self.inbound_neurons[0].value
		#weights = self.inbound_neurons[1].value
		#bias = self.inbound_neurons[2]
		#self.value = bias.value
		#for w, x in zip(inputs, weights):
		#self.value += x * w
		X = self.inbound_neurons[0].value
		W = self.inbound_neurons[1].value
		b = self.inbound_neurons[2].value
		self.value = np.dot(X, W) + b	

	def backward(self):
		self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_neurons}

		for n in self.outbound_neurons:
			grad_cost = n.gradients[self]

			#[self.inbound_neurons[0]] = inputs, [self.inbound_neurons[1]] = weights
			#[self.inbounb_neurons[2]] = bias
			#grad_cost 为Linear传出节点(outbound_neurons)传递进来的变化率

			#inputs的代价梯度 = grab_cost * weights 	
			self.gradients[self.inbound_neurons[0]] += np.dot(grad_cost, self.inbound_neurons[1].value.T)

			#weights的代价梯度 = grab_cost * inputs
			self.gradients[self.inbound_neurons[1]] += np.dot(self.inbound_neurons[0].value.T, grad_cost)

			#对于bias，Linear对bias求导恒为1,所以bias的代价梯度 = 1 * grab_cost
			self.gradients[self.inbound_neurons[2]] += np.sum(grad_cost, axis=0, keepdims=False)
			#axis = 0 表示队列求和

			#为何是 += ：因为在每一个节点将误差传递给每一个传出节点。于是在backpropagation时，要求出每一个节点
			#的误差，就要将每一份传递出去给传出节点的误差加起来。于是用 +=

			#区分Backpropagation和Gradient Descent是两个步骤，通过Backpropagation找到gradient，于是找到
			#变化方向。再通过Gradient Descent来最小化误差

class Sigmoid(Neuron):
	"""Sigmoid function"""
	def __init__(self, neuron):
		Neuron.__init__(self, [neuron])

	def _sigmoid(self, x):
		return 1. / (np.exp(-x) + 1.)
	
	def forward(self):
		input_value = self.inbound_neurons[0].value
		self.value = self._sigmoid(input_value)		

	def backward(self):
		self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_neurons}

		for n in self.outbound_neurons:
			grad_cost = n.gradients[self]
			sigmoid = self.value
			self.gradients[self.inbound_neurons[0]] += sigmoid * (1 - sigmoid) * grad_cost
class MSE(Neuron):
	def __init__(self, y, a):
		Neuron.__init__(self, [y, a])

	def forward(self):
		#维度调整
		y = self.inbound_neurons[0].value.reshape(-1, 1)
		a = self.inbound_neurons[1].value.reshape(-1, 1)

		self.m = self.inbound_neurons[0].value.shape[0]

		self.diff = y - a
		self.value = np.mean(self.diff**2)

	def backward(self):
		self.gradients[self.inbound_neurons[0]] = (2 / self.m) * self.diff
		self.gradients[self.inbound_neurons[1]] = (-2 / self.m) * self.diff
			
def topological_sort(feed_dict):
	input_neurons = [n for n in feed_dict.keys()]

	G = {}
	neurons = [n for n in input_neurons]
	while len(neurons) > 0:
		n = neurons.pop(0)
		if n not in G:
			G[n] = {'in': set(), 'out': set()}
		for m in n.outbound_neurons:
			if m not in G:
				G[m] = {'in':set(), 'out': set()}
			G[n]['out'].add(m)
			G[m]['in'].add(n)
			neurons.append(m)

	L = []
	S = set(input_neurons)
	while len(S) > 0:
		n = S.pop()
		#isinstance判断一个对象是否是可迭代对象
		if isinstance(n, Input):
			n.value = feed_dict[n]

		L.append(n)
		for m in n.outbound_neurons:
			G[n]['out'].remove(m)
			G[m]['in'].remove(n)
			#if no other incoming edges add to S
			if len(G[m]['in']) == 0:
				S.add(m)

	return L

def forward_pass(output_neuron, sorted_neurons):
	"""
	Performs a forward pass through a list of sorted neuron
	Arguments:
		'output_neuron':A neuron in the graph, should be the output neuron
		(has no outgoing edges)
		'sorted_neurons': a togologically sorted list of neuron
		神经元的拓扑分类列表
	"""		
	for n in sorted_neurons:
		n.forward()

	return output_neuron.value

def forward_and_backward(graph):
	"""
	通过排序列表(a list of sorted nodes)执行前向传播和反向传播

	Arguments: 'graph': The result of calling 'topological_sort'.
	拓扑排序结果
	"""
	#前向传播
	for n in graph:
		n.forward()
	#反向传播
	for n in graph[::-1]:
		n.backward()


def sgd_update(trainables, learning_rate = 1e-2):
	"""
	用SGD更新每个可训练的值
	'trainables': A list of 'Input' Nodes representing weights/biases. => trainalbes = [w1, b1, w2, b2]
	权重和偏差列表
	'learning_rate': 学习率
	"""
	for t in trainables:
		t.value -= t.gradients[t] * learning_rate



		