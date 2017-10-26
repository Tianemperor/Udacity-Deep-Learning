#coding:utf-8
import numpy as np

class Node(object):
	"""
	Base class for nodes in the network
	inbound_nodes:A list of nodes with edges into this node
	"""
	def __init__(self, inbound_nodes=[]):
		"""
		Node's constructor
		Sets properties that all nodes need
		"""
		self.inbound_nodes = inbound_nodes

		#节点最终值 The eventual value of this node.
		#Set by running the forward() method
		self.value = None

		#A list of nodes that this node outputs to.
		self.outbound_nodes = []

		self.gradients = {}

		#Set this node as an outbound node for all of this node's inouts
		for node in inbound_nodes:
			node.outbound_nodes.append(self)

	def forward(self):
		#每个节点需要定义自己节点的前向传播函数
		raise NotImplementedError

	def backward(self):
		#每个节点需要定义自己节点的反向传播函数
		raise NotImplementedError

class Linear(Node):
	"""
	Represents a node that performs a linear transform
	表示执行线性变换的节点
	"""
	def __init__(self, X，W, b):
		#权重和偏差
		Node.__init__(self, [X，W, b])

	def forward(self):
		#执行线性变换后的数值
		X = self.inbound_nodes[0].value
		W = self.inbound_nodes[1].value
		b = self.inbound_nodes[2].value
		self.value = np.dot(X, W) + b

	#def backward(self):
		#根据输出值计算梯度
		#初始化
		#self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

		#for n in self.outbound_nodes:
			#grad_cost = n.gradients[self]
			#self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
			#self.grad_cost[self.inbound_nodes[1]] += np.dot()







