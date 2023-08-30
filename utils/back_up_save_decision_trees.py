import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
pd.set_option('mode.chained_assignment', None)
from tqdm import tqdm
import random

def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class Decision_tree():
    """
    Initialize the Decision Tree.
    """
    def __init__(self, data: pd.DataFrame, target: str, depth: int, ratio: float, random_state: int = 1992, weights = None) -> None:
        self.data = data
        self.size = len(self.data)
        self.columns = data.columns.tolist()
        self.target = target
        self.splits = [('Sex', 'male')]
        self.depth = depth
        self.ratio = ratio
        self.random_state = random_state
        random.seed(random_state)  # Set random state for reproducibility
        self.weights = np.repeat(1/self.size, self.size) if weights == None else weights
        self.run_splits()

    def run_splits(self):
        while self.depth >= 1:
            self.new_split()
            self.depth -= 1
        

    def split_to_node(self, condition):
        node_left = self.data[condition].copy()
        node_right = self.data[~condition].copy()
        return node_left, node_right
    
    
    def select_columns(self):
        # Calculate best split for left and right child
        random.shuffle(self.columns)
        # Calculate the number of columns to select (10%)
        num_columns_to_select = int(len(self.columns) * self.ratio)

        # If the calculated number of columns is zero, select at least one column
        num_columns_to_select = max(num_columns_to_select, 1)
        
        return self.columns[:num_columns_to_select]


    def new_split(self):
        best_score= 0
        best_new_split = None
        for split in self.splits:
            # Go to the branch and look at left and right child
            if is_numeric( split[1]):
                node_left, node_right = self.split_to_node(self.data[split[0]] >= float(split[1]))
            else:
                node_left, node_right = self.split_to_node(self.data[split[0]] == split[1])
                
            # Select the proportion of the columns
            selected_columns = self.select_columns()
            
            # Find the best split for the nodes
            left_split, left_score = self.best_slit_for_node(node_left, selected_columns)
            right_split, right_score = self.best_slit_for_node(node_right, selected_columns)
            
            # Update best split
            temp_dict = {left_score : left_split, right_score : right_split,  best_score: best_new_split}
            max_key = max(temp_dict.keys())
            # Get the value associated with the maximum key
            best_score = max_key
            
        # Find best score
        new_split = temp_dict[best_score]
        if new_split == None:
            return
        self.splits.append(new_split)
        print(self.splits)


    def best_slit_for_node(self,data, columns):
        self.gini_values = {}
        for col  in columns:
            for val in self.data[col].unique():
                gini_weighted = self.split(data, col, val)
                self.gini_values[col + '_split_' + str(val)] = gini_weighted
        # Find the minimum value of the dictionary and update the self.data
        min_key = min(self.gini_values, key=self.gini_values.get)
        col, val = min_key.split('_split_')
        return (col, val), self.gini_values[min_key]


    def split(self, data, col, val):
        # Calculate probability for each node
        cls1 = data[data[col] == val]
        cls2 = data[data[col] != val]
        prob_list, n1, n2 = self.calculate_probability(cls1, cls2)
        
        # Calculate gini_index for each node
        g1, g2 = self.gini_index(prob_list)
        
        # Return weighted gini-impuritiny index
        return self.gini_impurity_for_split(g1, g2, n1, n2)


    def calculate_probability(self, cls1, cls2):
        ''''''
        prob_cls1 = []
        prob_cls2 = []
        n1 = len(cls1)
        n2 = len(cls2)
        for val in cls1[self.target].unique():
            temp_data = len(cls1[cls1[self.target] == val].copy())
            prob_cls1.append(temp_data / n1)

        for val in cls2[self.target].unique():
            value2 = len(cls2[cls2[self.target] == val].copy())
            prob_cls2.append(value2 / n2)
        return [prob_cls1, prob_cls2], n1, n2
        

    def gini(self, data):
        weighted_elements = np.sum([(element * weight_i) **2 for element, weight_i in zip(data, self.weights)])
        return 1 - weighted_elements / np.sum(self.weights)

    def gini_index(self, prob_list):
        g1 =  1 - self.gini(prob_list[0])
        g2 = 1 - self.gini(prob_list[1])ll
        return g1, g2
        

    def gini_impurity_for_split(self, g1, g2, n1, n2):
        return n1/(n1+n2) * g1 + n2/(n1+n2) * g2
        