import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
pd.set_option('mode.chained_assignment', None)
import uuid
from typing import List
import pdb

class ObjectiveFunction:
    """Class to calculate the Gini index for a given list of events."""
    def __init__(self, event_list: List[float]):
        self.event_list = event_list
        self.gini_index = self.calculate_gini_index()

    def calculate_gini_index(self) -> float:
        """Calculates and returns the Gini index."""
        total = sum(self.event_list)
        gini_index = 1.0
        for event in self.event_list:
            prob = event / total
            gini_index -= prob ** 2
        return gini_index

    def get_gini_index(self) -> float:
        """Returns the calculated Gini index."""
        return self.gini_index


class Splitter():
    '''This class returns all splits for a given data (categorical columns only)'''
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.split_list = self.create_splits()

    def create_splits(self):
        '''
        This class creates all possible splits of the data
        and returns a list of conditionals
        '''
        split_list = []
        for col in self.data.columns:
            for value in self.data[col].unique():
                split_list.append((col, value))
        return split_list

    def get_splits(self):
        """Getter of the different splits"""
        return self.split_list


class BestSplitFinder():
       def __init__(self,  data : pd.DataFrame, target : str):
              self.data = data
              self.target = target
              self.splitter = Splitter(self.data.drop(self.target, axis = 1)).get_splits()
              self.best_split = self.loop_through_splits()

       def loop_through_splits(self):
              """This function loops through all splits and find the one with the minimum gini index"""
              split_dictionary = {}
              for split in self.splitter:
                     number_of_events_in_this_split = self.calculate_target_counts(split)
                     objective = ObjectiveFunction(number_of_events_in_this_split)
                     split_dictionary[split] = objective.get_gini_index()

              # Find the key that corresponds to the minimum valuedd
              min_key = min(split_dictionary, key=split_dictionary.get)
              return min_key, split_dictionary.get(min_key, None)

       def calculate_target_counts(self, split):
              """Helps calculating the values that are target vs not target in a split"""
              col, val = split
              subset = self.data[self.data[col] == val]
              return subset[self.target].value_counts().tolist()


       def get_best_split(self):
              """Rreturns the best split featuer and value"""
              return self.best_split


class Node():
    def __init__(self, data, target):
        self.data = data
        self.target = target
        (self.col, self.val), self.gini_index = self.calculate_gini_index()

    
    def calculate_gini_index(self):
        best_split = BestSplitFinder(self.data, self.target)
        return best_split.get_best_split()

    def get_gini_index(self):
        return self.gini_index

    def get_col_and_val(self):
        return self.col, self.val

    def get_data(self):
        return self.data

    def get_child_boolean_mask(self, direciton):
        boolean_mask_updated =  self.data[self.col] == self.val
        return boolean_mask_updated if direciton == True else ~ boolean_mask_updated



class Prediction_Tree():
    def __init__(self, new_data, splits):
        self.new_data = new_data
        self.splits = splits

    def predict(self):
        data = self.new_data
        print(data)
        for key, item in self.splits.items():
             col, val = item.get_col_and_val()
             temp = item.get_data()
             print(col, val)
             print(temp[temp[col] == val])
             print('------------')


class AllSplits():
    def __init__(self, data, target, depth = 2):
        self.data = data
        self.target = target
        self.depth = depth
        self.splits = {}
        self.tree_splits = self.create_tree()

    
    def create_tree(self):
        boolean_mask = None
        splits = {}
        col_val = {}
        for _ in range(self.depth):
            if splits == {}:
                random_uuid = uuid.uuid4()  
                splits[random_uuid] =  Node(self.data, self.target)
                col_val[random_uuid] =  (None, None)
            else:
                # Loop through all the splits
                gini_index = {}
                for key, node in splits.items():
                    gini_index[key] = node.get_gini_index()

                # Find the split with the smallest gini index
                min_key = min(gini_index, key=gini_index.get)    
                # Get the data corresponding to the min_key
                current_data = splits[min_key].get_data()

                # create two new more splits and add it to the dictionary
                for direction in [True, False]:
                    boolean_mask_direction = splits[min_key].get_child_boolean_mask(direction)
                    node_data = current_data[boolean_mask_direction]
                    if node_data.empty:
                        return splits, col_val

                    # Create new node and values
                    node_new =  Node(node_data, self.target)
                    col, val = node_new.get_col_and_val()

                    # Save the split and the columns and values of the split
                    rand_uuid = uuid.uuid4()
                    splits[rand_uuid] = node_new
                    col_val[rand_uuid] = (col, val)

                # delete old split
                del splits[min_key]
                del col_val[min_key]

        return splits, col_val
        
    def get_tree_splits(self):
        return self.tree_splits