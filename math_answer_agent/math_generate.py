import math
import numpy as np
import pandas as pd

class modular_math_generator(object):
    def __init__(self):
        pass

    def judge_int(self,float_num):
        if float(int(float_num)) == float_num:
            return True
        else:
            return False
            
    def get_truth(self, n1, n2, n3):
        if self.judge_int(float(n1 - n2)/float(n3)) == True:
            return 1  
        else:
            return 0
    
    def math_digits_generator(self):
        num_1_unit = np.random.randint(1, 10) 
        num_1_ten = np.random.randint(2, 10) 
        num_1 = num_1_ten * 10 + num_1_unit             
    
        num_2_unit = np.random.randint(1,10)
        num_2_ten = np.random.randint(1, num_1_ten)
        num_2 = num_2_ten * 10 + num_2_unit

        num_3 = np.random.randint(3, 10)
        return num_1, num_2, num_3

    def math_question_generator(self):
        target_truth = np.random.randint(0, 2)
        self.num_1, self.num_2, self.num_3 = self.math_digits_generator()
        self.truth = self.get_truth(self.num_1, self.num_2, self.num_3)

        while self.truth != target_truth:
            self.num_1, self.num_2, self.num_3 = self.math_digits_generator()
            self.truth = self.get_truth(self.num_1, self.num_2, self.num_3)
        
        self.math_question_str = str(self.num_1) + " ≡ " + str(self.num_2) + " ( mod " + str(self.num_3) + " )"

        return self.num_1, self.num_2, self.num_3, self.truth, self.math_question_str

def shuffle_dataset(data, seed = 4):
    np.random.seed(seed)
    rng = np.random.default_rng()
    numbers = rng.choice(data.shape[0], size=data.shape[0], replace=False)
    new_data = data[numbers].copy()
    return new_data


def generate_dataset(train_test_rate = 0.8, train_filename = 'train_numeric.csv', test_filename = 'test_numeric.csv', answer_type = 'numeric'):
    math_generator = modular_math_generator()
    all_data = []
    for num_1_unit in range(1,10):
        for num_1_ten in range(2,10):
            for num_2_unit in range(1,10):
                for num_2_ten in range(1, num_1_ten):
                    for num_3 in range(3, 10):
                        num_1 = int(num_1_ten * 10 + num_1_unit)             
                        num_2 = int(num_2_ten * 10 + num_2_unit)
                        assert answer_type == 'binary' or answer_type == 'numeric'
                        if answer_type == 'binary':
                            math_truth = math_generator.get_truth(num_1, num_2, num_3)
                        else:
                            math_truth = (num_1 - num_2) % int(num_3) 
                        all_data.append([num_1, num_2, num_3, math_truth])
    all_data_arr = np.array(all_data)
    dataset_len = len(all_data_arr)
    all_data_arr_shuffled = shuffle_dataset(all_data_arr)

    train_data, test_data = all_data_arr_shuffled[:int(dataset_len*train_test_rate)], all_data_arr_shuffled[int(dataset_len*train_test_rate):]
    print('train shape: ', train_data.shape, ', test shape: ', test_data.shape)

    csv_head = ['num_1', 'num_2', 'num_3', 'answer']
    train_table = pd.DataFrame(train_data,columns=csv_head)
    test_table = pd.DataFrame(test_data,columns=csv_head)
    train_table.to_csv(train_filename, index=False)
    test_table.to_csv(test_filename, index=False)

generate_dataset()