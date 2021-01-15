import argparse
import random
import numpy as np
import time
import concurrent.futures

from dataset import Dataset
from decision_stump import BranchingDecisionStump


class DecisionTree(object):
    depth = 0
    leaf_num = 0

    def __init__(self, depth=1, verbose=True):
        self.branch = BranchingDecisionStump()
        self.depth = depth
        self.is_root = (self.depth == 1)
        self.subtrees = {1: None, 2:None}
        self.is_leaf = None
        self.base_hypothesis = None
        
        self.verbose = verbose
        
        self.terminate = False
        

    def train(self, data_x, data_y):
        if self.is_root and self.verbose:
            print('===== START TRAINING DECISION TREE =====')

        # check if reach terminate
        (unique, counts) = np.unique(data_y, return_counts=True)
        if unique.shape[0] == 1:
            self.terminate = True

        if self.terminate:
            self.is_leaf = True
            DecisionTree.leaf_num += 1
            self.base_hypothesis = unique[0]
            if self.verbose:
                print("Ternimate! leaf_num: ", DecisionTree.leaf_num)

        elif not self.terminate:
            self.is_leaf = False
            DecisionTree.depth = max(DecisionTree.depth, self.depth+1)
            if self.verbose:
                print("Depth: ", self.depth)

            one_data_x, one_data_y, two_data_x, two_data_y = self.branch.fit(data_x, data_y, self.verbose)

            self.subtrees[1] = DecisionTree(self.depth+1, verbose=self.verbose)
            self.subtrees[1].train(one_data_x, one_data_y)
            self.subtrees[2] = DecisionTree(self.depth+1, verbose=self.verbose)
            self.subtrees[2].train(two_data_x, two_data_y)
        
        if self.is_root and self.verbose:
            print('\n===== TRAINING ENDED =====')
            print('Depth: ', DecisionTree.depth)
            print('Leafs: ', DecisionTree.leaf_num)
            self.whole_tree_depth = DecisionTree.depth
            self.whole_tree_leaf_num = DecisionTree.leaf_num
            DecisionTree.depth, DecisionTree.leaf_num = 0, 0


    def predict(self, data_x):
        """ Predict one sample

        Input:
            :input: 1-D numpy array
                size = (the number of features,)
        """
        assert self.is_leaf is not None

        if self.is_leaf:
            return self.base_hypothesis
        else:
            return self.subtrees[self.branch.predict(data_x)].predict(data_x)


    def predict_batch(self, data_x):
        pred_y = []
        for x in data_x :
            pred_y.append(self.predict(x))

        return np.array(pred_y)


class RandomForest(object):
    def __init__(self, n_trees, seed=1126):
        self.n_trees = n_trees
        self.trees = []
        self.seed = seed
        self.used_sample_per_tree = {}
        self.used_tree_per_sample = {}

        self.all_data_size = 0
        self.bagging_data_size = 0

    @staticmethod
    def unique_index(index_list):
        index_list = np.array(index_list)
        (unique, counts) = np.unique(index_list, return_counts=True)
        return unique


    def train(self, data_x, data_y, sample_rate=0.5):
        print('===== START TRAINING RANDOM FOREST =====')
        self.all_data_size = data_x.shape[0]
        self.bagging_data_size = self.all_data_size * sample_rate
        for tree_idx in range(self.n_trees):
            dt = DecisionTree(depth=1, verbose=False)
            bagging_data_x, bagging_data_y, bagging_idx = Dataset.couple_bagging_for_training(data_x, data_y, random_seed=self.seed, sample_rate=sample_rate)
            dt.train(bagging_data_x, bagging_data_y)
            self.trees.append(dt)
            self.used_sample_per_tree[tree_idx] = self.unique_index(bagging_idx)
            print("grown trees: {}/{}".format(tree_idx+1, self.n_trees))
        
        self.find_all_tree_use_each_sample()
        print('===== TRAINING ENDED =====')
    

    def train_multiprocessing(self, data_x, data_y, sample_rate=0.5, workers=4):
        """ Still has some bug
        """
        print('===== START TRAINING RANDOM FOREST =====')
        self.all_data_size = data_x.shape[0]
        self.bagging_data_size = self.all_data_size * sample_rate

        for tree_idx in range(self.n_trees):
            dt = DecisionTree(depth=1, verbose=False)
            bagging_data_x, bagging_data_y, bagging_idx = Dataset.couple_bagging_for_training(data_x, data_y, random_seed=self.seed, sample_rate=sample_rate)
            start_time = time.time()
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor: 
                executor.submit(dt.train, bagging_data_x, bagging_data_y)
            training_used_time_per_tree = time.time() - start_time
            self.trees.append(dt)
            self.used_sample_per_tree[tree_idx] = self.unique_index(bagging_idx)
            print("grown trees: {}/{}; elapsed time: {} sec".format(tree_idx+1, self.n_trees, training_used_time_per_tree))
        
        self.find_all_tree_use_each_sample()
        print('===== TRAINING ENDED =====')
    

    def predict(self, data_x, specified_tree_idx=[]):
        pred_ys = []
        if not specified_tree_idx:
            for tree in self.trees:
                pred_ys.append(tree.predict_batch(data_x))
        else:
            for tree_idx in specified_tree_idx:
                pred_ys.append(self.trees[tree_idx].predict_batch(data_x))
        
        pred_y = np.asarray(pred_ys).mean(axis=0)
        for y_idx in range(pred_y.shape[0]):
            pred_y[y_idx] = 1.0 if pred_y[y_idx] > 0. else -1.0
        
        return pred_y

    def find_all_tree_use_each_sample(self):
        for sample_idx in range(self.all_data_size):
            tree_idx_list = []
            for tree_idx in range(self.n_trees):
                if sample_idx in self.used_sample_per_tree[tree_idx]:
                    tree_idx_list.append(tree_idx)
            self.used_tree_per_sample[sample_idx] = tree_idx_list


if __name__  == "__main__":
    dataset = Dataset()
    train_x, train_y = dataset.get_data('train')
    test_x, test_y = dataset.get_data('test')

    from sklearn.metrics import accuracy_score

    '''
    # 14
    print('Problem 14')
    dt = DecisionTree(verbose=True)
    dt.train(train_x, train_y)
    pred_y = dt.predict_batch(test_x)
    print("Eout: ", 1-accuracy_score(test_y, pred_y))
    print()
    print()
    #'''
    
    rf = RandomForest(n_trees=5)
    start_time = time.time()
    #with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor: 
    #    executor.submit(rf.train, train_x, train_y)
    rf.train(train_x, train_y)
    training_used_time = time.time() - start_time
    print("=== Training Time: {} sec".format(training_used_time))
    
    
    # 15
    print('Problem 15')
    eouts = []
    for tree in rf.trees:
        pred_y = tree.predict_batch(test_x)
        eouts.append(1-accuracy_score(test_y, pred_y))
    print("Eout_mean: ", sum(eouts)/len(eouts))
    print()
    print()


    # 16
    print('Problem 16')
    pred_y = rf.predict(train_x)
    print("Ein: ", 1-accuracy_score(train_y, pred_y))
    print()
    print()


    # 17
    print('Problem 17')
    pred_y = rf.predict(test_x)
    print("Eout: ", 1-accuracy_score(test_y, pred_y))
    print()
    print()


    #'''
    # 18
    err = []
    print('Problem 18')
    for sample_idx in range(train_x.shape[0]):
        if len(rf.used_tree_per_sample[sample_idx]) == 0:
            err.append(-1)
        else:
            sample = train_x[sample_idx, :]
            sample = sample[np.newaxis, :]
            e = rf.predict(sample, specified_tree_idx=rf.used_tree_per_sample[sample_idx])
            err.append((0 if e[0]==train_y[sample_idx] else 1))
    print("Eoob: ", sum(err)/len(err))
    print()
    print()
    #'''




