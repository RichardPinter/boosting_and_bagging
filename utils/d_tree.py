import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('mode.chained_assignment', None)

class Decision_tree():


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


    def best_slit_for_node(self,data, columns):
        gini_values = {}
        for col  in columns:
            for val in self.data[col].unique():
                gini_weighted = self.split(data, col, val)
                gini_values[col + '_split_' + str(val)] = gini_weighted
        # Find the minimum value of the dictionary and update the self.data
        min_key = min(gini_values, key=gini_values.get)
        col, val = min_key.split('_split_')
        return (col, val), gini_values[min_key]


    
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
            value_1 = len(cls1[cls1[self.target] == val].copy())
            prob_cls1.append(value_1.index())

        for val in cls2[self.target].unique():
            value_2 = len(cls2[cls2[self.target] == val].copy())
            prob_cls2.append(value_2.index())
        return [prob_cls1, prob_cls2], n1, n2
        

    def gini(self, data):
        print(data)
        weighted_elements = np.sum([(element * weight_i) **2 for element, weight_i in zip(data, self.weights)])
        return 1 - weighted_elements / np.sum(self.weights)


    def gini_index(self, prob_list):
        g1 =  1 - self.gini(prob_list[0])
        g2 = 1 - self.gini(prob_list[1])
        return g1, g2
        

    def gini_impurity_for_split(self, g1, g2, n1, n2):
        return n1/(n1+n2) * g1 + n2/(n1+n2) * g2






#######################################################################################
    def create_tree(self):
        """This function creates the tree"""
        splits = []
        combined_mask = None
        for idx, _ in enumerate(range(self.depth)):
            if splits == []:
                best_split = BestSplitFinder(self.data, self.target, combined_mask)
                min_key, gini_value = best_split.get_best_split()
            else:
                best_splits_list = {}
                for split in splits:
                    best_split_left = BestSplitFinder(self.data, self.target, ~split)
                    best_split_right = BestSplitFinder(self.data, self.target, split)
                for best_split in best_splits_list:
                    (col, val), gini_value = best_split.get_best_split()
                    split_dictionary[(col, val)] = gini_value
                # Find the key that corresponds to the minimum valuedd
                min_key = min(split_dictionary, key=split_dictionary.get)
            col, val = min_key
            print(min_key)
            current_mask = (self.data[col] == val).values
            print(len(current_mask))
            # Update the combined mask
            combined_mask = current_mask
            # if combined_mask is None:
            # else:
            #     combined_mask &= current_mask  # Combine with the existing mask using bitwise AND
            splits.append(combined_mask)
        return splits