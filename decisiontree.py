from __future__ import division
import math
import operator
import time
import random
import copy
from collections import Counter

############################################################################

'''
TreeNode class represents a decision tree which can be used to predict result.
It takes user's input and make a decision which branch seperated by target attribute
'''
class TreeNode():
	# @brief Predict reuslts according to user's input
	# @param e - a dictionary which keys are all kind of attributes paired with values
	def predict(self, e):
		# return result when it already reaches the end of tree
		if isinstance(self, TreeLeaf):
			return self.result
		else:
			# Decide which path to go according to the comparison result between e[self.attr] and split value
			if self.numeric:
				if e[self.attr] <= self.splitval and '<=' in self.branches:
					return self.branches['<='].predict(e)
				elif '>' in self.branches:
					return self.branches['>'].predict(e)
				else:
					return '0'
			else:
				try:    
					out = self.branches[e[self.attr]].predict(e)
				except:
					return '0'
				return out
	
	# @brief Build all valid path in the tree
	# @param path - empty string which used to build all valid paths
	def disjunctive_nf(self, path):
		# If leafNode's value is not equal to 1 which means it is not a valid path then
		# return false or append valid path to results
		if isinstance(self, TreeLeaf):
			if self.result == '1':
				print "("+ str(path) + ") OR"
				return path
			else:
				return False
				
		# Recursive find path for branches
		else:
			for branch_label, branch in self.branches.iteritems():
				if self.numeric:
					clause = self.attr_name + " " + branch_label + " "+ str(self.splitval)
				else: 
					clause = self.attr_name + " is " + branch_label
				new_path = path + [clause]
				branch.disjunctive_nf(new_path)
		return path
	
	# @brief List all nodes in the treeNode
	# @param nodes - an array which contains all nodes in the tree
	def list_nodes(self, nodes):
		# If it reaches the leaf, append it to the nodes array
		if isinstance(self, TreeLeaf):
			nodes.append(self)
			return nodes
		nodes.append(self)
		for branch_label, branch in self.branches.iteritems():
			nodes = branch.list_nodes(nodes)
		return nodes

############################################################################

'''
The TreeLeaf class represents a leafNode in a tree
and inheritate all methods from TreeNode class
'''
class TreeLeaf(TreeNode):
	# @brief Initialize TreeLeaf object
	def __init__(self, target_class):
		self.result = target_class

	# @brief Print information of TreeLeaf object
	def __repr__(self):
		return "This is a TreeLeaf with result: {0}".format(self.result)

	# @brief Update state variables when this object becomes a parent node
	def toFork(self):
		self.__class__ = TreeFork
		self.result = None
############################################################################

'''
The TreeFork class represents a normalNode(not leaf) in a tree
and inheritate all methods from TreeNode class
'''
class TreeFork(TreeNode):
	# @brief Initialize TreeFork object
	# @param attr_arr - an array contains all information of a node
	def __init__(self, attr_arr):
		self.attr =  attr_arr[0]
		self.splitval =  attr_arr[1]
		self.numeric = attr_arr[2]
		self.attr_name = attr_arr[3]
		self.mode = attr_arr[4]
		self.branches = {}

	# @brief Update state variables when this object becomes a leaf node
	# @brief target - an integer value represents the reuslt
	def toLeaf(self, target):
		self.__class__ = TreeLeaf
		self.result = target
		
	# @brief Add a branch to the this node
	# @param val - key value represents a new entry for the branch
	# @param subtree - new branch added to this node
	def add_branch(self, val, subtree, default):
		self.branches[val] = subtree
	
	# @brief Print information of this node
	def __repr__(self):
		return "\nThis node is a fork on {0}, with {1} branches.\nMost instances in it are {2}.".format(self.attr_name,len(self.branches),self.mode)
############################################################################

'''
The Dataset class is used to initialize raw data
'''
class Dataset:
	# @brief Initialize dataset and handle missing attribute
	# @brief filename - a string represents the name of the file
	def __init__(self, filename, test=False):
		self.filename = filename
		self.arrange_data()
		#Fill in missing attribute to help analyze data
		if not test:
			self.fill_missing()

	# @brief Handle raw dataset to make preparation for later analysis
	def arrange_data(self):
		with open(self.filename) as f:
			original_text = f.read()
			self.numeric_attrs = [True,True,True,False,False,False,False,False,True,True,True,True,False]

			# split data by instances... if last line is empty, delete it
			self.instances = [line.split(',') for line in original_text.split("\n")]

			if self.instances[-1]==['']:
				del self.instances[-1]

			# set header row as attributes, remove it from dataset
			self.attr_names = self.instances.pop(0)

			# Change strings to int where necessary
			for instance in self.instances:
				for a in range(len(self.numeric_attrs)):
					if instance[a] == '?':
						continue
					if self.numeric_attrs[a]:
						instance[a] = int(instance[a])

			# self.values = [[e[attr] for e in self.examples] for attr in self.attributes]

	# @brief Fill in the missing attribute value of dataset
	def fill_missing(self):
		# 0 represents invalid path and 1 represents valid path
		groups = [0, 1]
		fill_values = 2*[[None] * len(self.attr_names)]
		for g in groups:
			group_instances = [e for e in self.instances if e[-1] == str(groups[g])]
			# Allocate group for each attribute in the fill_values list
			for attr in range(len(self.attr_names[:-1])):
				if self.numeric_attrs[attr]:
					not_missing = [e[attr] for e in group_instances if e[attr] != '?']      
					fill_values[g][attr] = int(sum(not_missing) / len(not_missing))
				else:
					nominal_vals = [e[attr] for e in group_instances]
					fill_values[g][attr] = Counter(nominal_vals).most_common()[0][0]
		
		# Updates each instance/raw attribute value in the data set
		for instance in self.instances:
			for attr in range(len(self.attr_names)):
				if instance[attr]=='?' and instance[-1]=='0':
					instance[attr] = fill_values[0][attr]
				elif instance[attr]=='?' and instance[-1]=='1':
					instance[attr] = fill_values[1][attr]
		return True

############################################################################

# @brief Count the number of this attributes occurs in the data set
# @param examples - all instances in the training data set
# @param attributes - an array of all attributes
# @param target - target index
def getFrequencies(examples, attributes, target):
	frequencies = {}
	#find target in data
	a = attributes.index(target)
	#calculate frequency of values in target attr
	for row in examples:
		if row[a] in frequencies:
			frequencies[row[a]] += 1 
		else:
			frequencies[row[a]] = 1
	return frequencies

############################################################################

# @brief Calculates the entropy of the given data set for the target attribute.
# @param examples - all instances in the training data set
# @param attributes - an array of all attributes
# @param target - target index
def entropy(examples, attributes, target):

	dataEntropy = 0.0
	frequencies = getFrequencies(examples, attributes, target)
	# Calculate the entropy of the data for the target attr
	for freq in frequencies.values():
		dataEntropy += (-freq/len(examples)) * math.log(freq/len(examples), 2) 
	return dataEntropy

############################################################################

# @brief Calculates the information gain(reduction in entropy) that would 
# result by splitting the data on the chosen attribute(for uncontinuous
# string)
# @param examples - all instances in the training data set
# @param attributes - an array of all attributes
# @param attr - target index
# @param targetAttr - target attribute
def gain(examples, attributes, attr, targetAttr):
	numeric_attrs = [True,True,True,False,False,False,False,False,True,True,True,True,False]

	currentEntropy = entropy(examples, attributes, targetAttr)
	subsetEntropy = 0.0
	i = attributes.index(attr)
	best = 0

	# Calculate entropy when it's value is true in numeric_attrs array
	if numeric_attrs[i]:
		order = sorted(examples, key=operator.itemgetter(i))
		subsetEntropy = currentEntropy
		for j in range(len(order)):
			if j==0 or j == (len(order)-1) or order[j][-1]==order[j+1][-1]:
				continue

			currentSplitEntropy = 0.0
			subsets = [order[0:j], order[j+1:]]
			
			# Calculate the total entropy
			for subset in subsets:
				setProb = len(subset)/len(order)
				currentSplitEntropy += setProb*entropy(subset, attributes, targetAttr)

			if currentSplitEntropy < subsetEntropy:
				best = order[j][i]
				subsetEntropy = currentSplitEntropy
				
	# Calculate entropy when it's value is false in numeric_attrs array
	else:
		valFrequency = getFrequencies(examples, attributes, attr)

		for val, freq in valFrequency.iteritems():
			valProbability =  freq / sum(valFrequency.values())
			dataSubset     = [entry for entry in examples if entry[i] == val]
			subsetEntropy += valProbability * entropy(dataSubset, attributes, targetAttr)
	
	return [(currentEntropy - subsetEntropy),best]

############################################################################

# @brief Select an attribute to decide which branch or paths it goes in the tree
# @param examples - all instances in the training data set
# @param target - target index
def selectAttr(examples, attributes, target):
	best = False
	bestCut = None
	maxGain = 0
	for a in attributes[:-1]:
		newGain, cut_at = gain(examples, attributes, a, target) 
		# Select the attribute with max gain value
		if newGain>maxGain:
			maxGain = newGain
			best = attributes.index(a)
			bestCut = cut_at
	return [best, bestCut]
############################################################################

# @brief Return true if all classes in the instance matches first_class
# @param examples - all instances in the training data set
def one_class(examples):
	first_class = examples[0][-1]
	for e in examples:
		if e[-1]!=first_class:
			return False
	return True
############################################################################

# @brief Change filter attributes
# @param examples - all instances in the training data set
# @param index - target index
def mode(examples, index):  
	L = [e[index] for e in examples]
	return Counter(L).most_common()[0][0]

############################################################################

# @brief split one decision tree into several smaller decision trees
# @param examples - all instances in the training data set
# @param splitval - seperable integer value
def splitTree(examples, attr, splitval):
	numeric_attrs = [True,True,True,False,False,False,False,False,True,True,True,True,False]
	isNum = numeric_attrs[attr]
	positive_count = 0

	# Seperate the leaf node
	if isNum:
		subsets = {'<=': [], 
					'>': []}
		for row in examples:
			if row[-1]=='1':
				positive_count += 1
			if row[attr]<=splitval:
				subsets['<='].append(row)
			elif row[attr]>splitval:
				subsets['>'].append(row)
				
	# Seperate the TreeFork node
	else:
		subsets = {}
		for row in examples:
			if row[-1]=='1':
				positive_count += 1
			if row[attr] in subsets:
				subsets[row[attr]].append(row)
			else:
				subsets[row[attr]] = [row]
	negative_count = len(examples)-positive_count
	if positive_count > negative_count:
		majority = '1'
	else:
		majority = '0'

	out = {"splitOn": splitval, "branches": subsets, "numeric": isNum, "mode": majority}
	return out

############################################################################

# @brief Build the decisision tree used for prediction
# @param examples - all instances in the data set
# @param attributes - an array of all attributes
# @param target - target index
# @param interation - times of max iteration
def learn_decision_tree(examples, attributes, default, target, iteration):
	iteration += 1

	# Split tree when number of interation reaches the maximum value
	if iteration > 10:
		return TreeLeaf(default)
	# Set tree to leafNdode when all instances in the examples are used
	if not examples:
		tree = TreeLeaf(default)
	# one_class means we don't need to choose target attribute
	elif one_class(examples):
		tree = TreeLeaf(examples[0][-1])
	else:
		best_attr = selectAttr(examples, attributes, target)
		if best_attr is False:
			tree = TreeLeaf(default)

		else:
			# new decision tree with root test *best_attr*
			split_examples = splitTree(examples, best_attr[0], best_attr[1])
			best_attr.append(split_examples['numeric'])
			best_attr.append(attributes[best_attr[0]])
			best_attr.append(split_examples["mode"])
			tree = TreeFork(best_attr)
			for branch_lab, branch_examples in split_examples['branches'].iteritems():
				if not branch_examples:
					break
				sub_default = mode(branch_examples, -1)
				subtree = learn_decision_tree(branch_examples, attributes, sub_default, target, iteration)
				tree.add_branch(branch_lab, subtree, sub_default)
	return tree

############################################################################

# @brief Calculate the accuracy of the decision tree
# @param examples - all instances in the data set
# @param dt - decision tree
def tree_accuracy(examples, dt):
	count = 0
	correct_predictions = 0
	for row in examples:
		count += 1
		pred_val = dt.predict(row)
		if row[-1]==pred_val:
			correct_predictions+=1
	accuracy = 100*correct_predictions/len(examples)
	return accuracy

###########################################################################

# @brief Test the decision tree using the testing data set
# @param examples - all instances in the data set
# @param dt - decision tree
def test_tree(examples, dt):
	for row in examples:
		row[-1] = dt.predict(row)
	return examples
############################################################################

# @brief Delete useless branch in the decision tree
# @param tree - decision tree
# @param nodes - tree node which is going to delete
# @param validation_examples - testing data set
# @param old_acc - accuracy before pruning
def prune_tree(tree, nodes, validation_examples, old_acc):
	percent_to_try = 0.2
	nodes = random.sample(nodes, int(percent_to_try*(len(nodes))))
	reduced_by = 1000
	# Prune the tree until it can not increase the accuracy
	while reduced_by >0:
		reduction = []
		# Go through all nodes to check which nodes can be deleted
		for n in nodes:
			if isinstance(n, TreeLeaf):
				nodes.pop(nodes.index(n))
				continue
			else:
				target_class = n.mode
				n.toLeaf(target_class)
				new_acc = tree_accuracy(validation_examples, tree)
				diff = new_acc - old_acc
				reduction.append(diff)
				n.toFork()
		# Delete all useless nodes listed in the reduction array
		if reduction != []:
			max_red_at = reduction.index(max(reduction))
			if isinstance(nodes[max_red_at], TreeFork):
				nodes[max_red_at].toLeaf(nodes[max_red_at].mode)
			nodes.pop(max_red_at)
			reduced_by = max(reduction)
			old_acc = tree_accuracy(validation_examples, tree)
		# If there is no node can be deleted, break out the loop
		else:
			reduced_by = 0

	print "The new accuracy is: " + str(new_acc) + "%"
	return [tree, new_acc]

############################################################################

# @brief The main method of the whole project
def main():
	# Calculate the time for building trees and predicting reuslts
	# Information for analysis
	now = time.time()
	target = "group"
	train_filename = "train.csv"
	validation_filename = "validate.csv"
	test_filename = "test.csv"
	train_data = Dataset(train_filename)
	validation_data = Dataset(validation_filename)
	test_data = Dataset(test_filename, True)
	
	#build tree
	default = mode(train_data.instances, -1)
	print "\nTime to learn this tree..."
	learned_tree = learn_decision_tree(train_data.instances, train_data.attr_names, default, target, 0)
	
	print "Trained on " +str(train_filename)+ ":"
	train_accuracy = tree_accuracy(train_data.instances, learned_tree)
	# Validate tree
	print "Training Accuracy= " + str(train_accuracy) + "%"

	validation_accuracy = tree_accuracy(validation_data.instances, learned_tree)
	print "Validation Accuracy= " + str(validation_accuracy) + "%"
	print "\nDISJUNCTIVE NORMAL FORM PRE PRUNING"
	dnf = learned_tree.disjunctive_nf([])
	nodes = learned_tree.list_nodes([])
	print "We have " + str(len(nodes)) + " nodes!"

	prePruningTime = time.time() - now	
	print "Pre-pruning Runtime = " + str(prePruningTime) + "\n"
	# Prune tree
	pruned_learned_tree = prune_tree(learned_tree, nodes, validation_data.instances, validation_accuracy)

	print "\nDISJUNCTIVE NORMAL FORM POST PRUNING"
	dnf_pruned = pruned_learned_tree[0].disjunctive_nf([])
	nodes_pruned = pruned_learned_tree[0].list_nodes([])
	print "We have " + str(len(nodes_pruned)) + " nodes!"
	
	print "Now this is the test set!"
	tested_set = test_tree(test_data.instances, pruned_learned_tree[0])
	print tested_set

	# Calculate the total processing time
	totalTime = time.time() - now
	print "Final Stats:"
	print "Runtime = " + str(totalTime) + "\n"
	print "Train accuracy: " + str(train_accuracy)
	print "Validation pre-pruning accuracy: " + str(validation_accuracy)
	print "Validation post-pruning accuracy: " + str(pruned_learned_tree[1])
	print "Pre-pruning tree size: " + str(len(nodes))
	print "Post-pruning tree size: " + str(len(nodes_pruned))

if __name__ == "__main__":
	main()