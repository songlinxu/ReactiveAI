from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(4)

import pickle, os, sys, time 
from joblib import dump, load
import numpy as np
from numpy.random import seed
seed(4)
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats

class CharacterTable:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)

def remove_outlier(raw_data, target_axis):
    target_axis_data = raw_data[:,target_axis:target_axis+1]
    outlier_axis_data = [y for stat in boxplot_stats(target_axis_data) for y in stat['fliers']]
    outlier_index = np.isin(target_axis_data,np.array(outlier_axis_data)) == False
    outlier_slices = []
    for oi, va_inx in enumerate(outlier_index):
        if va_inx[0] == True:
            outlier_slices.append(oi)
    new_data = np.take(raw_data, outlier_slices, axis=0)
    return new_data


class answer_classifier_resptime_regressor():
    def __init__(self, lstm_model_path, MAXLEN, chars):
        # raw_dataset_path
        self.math_answer_agent_path = lstm_model_path
        self.MAXLEN = MAXLEN
        self.chars = chars
        self.ctable = CharacterTable(chars)
        model_accuracy = keras.models.load_model(lstm_model_path)
        self.model_accuracy_exp = keras.Model(inputs = model_accuracy.input, outputs = model_accuracy.get_layer('lstm_1').output)
        

    def data_prepare(self, select_type, user_id, input_dataset_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 0.8, max_time = 10, ex_out = True, group = 'none', vis = True, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = False, math_encode = 2, add_day = False, norm = False, dataset_type = 'train', user_list = None):
        self.user_id = user_id
        model_accuracy = keras.models.load_model(model_accuracy_path)
        model_accuracy_exp = keras.Model(inputs = model_accuracy.input, outputs = model_accuracy.get_layer('lstm_1').output)

        # user_id,group,day,session,question_id,num_1,num_2,num_3,feedback,answer,accuracy,resptime,focus,focustime,anxiety,anxietytime
        data_set = pd.read_csv(input_dataset_path)
        assert select_type == 'group' or select_type == 'all' or select_type == 'user' or select_type == 'user_list'
        if select_type == 'group':
            data_set_select = data_set[(data_set['group']==group) & (data_set['resptime']<max_time)]
        elif select_type == 'all':
            data_set_select = data_set[(data_set['resptime']<max_time)]
        elif select_type == 'user':
            data_set_select = data_set[(data_set['user_id']==user_id) & (data_set['resptime']<max_time)]
        else:
            data_set_select = data_set[(data_set['user_id'].isin(user_list)) & (data_set['resptime']<max_time)]
        data_set_arr = np.array(data_set_select)
        print('data_set_arr.shape: ', data_set_arr.shape)
        # remove outlier
        if ex_out == True:
            data_set_filtered_arr = remove_outlier(data_set_arr, target_axis=11)
        else:
            data_set_filtered_arr = data_set_arr

        current_head = data_set.columns.values
        data_set = pd.DataFrame(data_set_filtered_arr, columns = current_head)
        questions, expected_resptime, expected_answer, ques_ids, feedbacks, num_1_all, num_2_all, num_3_all, day_ids = [], [], [], [], [], [], [], [], []
        for i in range(len(data_set)):
            sample = data_set.loc[i]
            num_1, num_2, num_3, resptime, answer, question_id, feedback, day_id = int(sample['num_1']), int(sample['num_2']), int(sample['num_3']), sample['resptime'], sample['answer'], sample['question_id'], sample['feedback'], sample['day']
            day_id = 0 if day_id == 'd1' else 1
            questions.append(str(num_1) + '≡' + str(num_2) + '(mod' + str(num_3) + ')')
            expected_resptime.append(resptime)
            expected_answer.append(answer)
            ques_ids.append(question_id)
            feedbacks.append(feedback)
            num_1_all.append(num_1)
            num_2_all.append(num_2)
            num_3_all.append(num_3)
            day_ids.append(day_id)
        # Vectorize the data
        ctable = CharacterTable(chars)
        x_math = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
        for i, sentence in enumerate(questions):
            x_math[i] = ctable.encode(sentence, MAXLEN)
        x_out = model_accuracy_exp.predict(x_math)

        if math_encode == 2:
            x = x_out.reshape((x_out.shape[0],x_out.shape[2]))
        elif math_encode == 1:
            x = x_math.reshape((len(questions),MAXLEN * len(chars)))
        else:
            x = np.concatenate((np.array(num_1_all).reshape((len(num_1_all),1)), np.array(num_2_all).reshape((len(num_2_all),1)), np.array(num_3_all).reshape((len(num_3_all),1))),axis=1)

        # PCA to lower dimension
        if pca_flag == True:
            pca = PCA(n_components=pca_num)
            x = StandardScaler().fit_transform(x)
            x = pca.fit_transform(x)

        # plt.scatter(x[:,0:1], x[:,1:2], c=expected)
        # plt.show()

        # add question id
        x = np.concatenate((x,np.array(ques_ids).reshape((len(ques_ids),1))),axis=1)
        if withfeedback == True:
            x = np.concatenate((x,np.array(feedbacks).reshape((len(feedbacks),1))),axis=1)
        if add_day == True:
            x = np.concatenate((x,np.array(day_ids).reshape((len(day_ids),1))),axis=1)
        if norm == True:
            assert dataset_type == 'train' or dataset_type == 'test'
            if dataset_type == 'train':
                self.scaler = StandardScaler()
                self.scaler.fit(x)
                x = self.scaler.transform(x)
            else:
                x = self.scaler.transform(x)

        y_resptime = np.array(expected_resptime)
        y_answer = np.array(expected_answer)
        if y_scale_resptime == True:
            self.y_resptime_min, self.y_resptime_max = np.min(y_resptime), np.max(y_resptime)
            y_resptime = (y_resptime-self.y_resptime_min)/(self.y_resptime_max-self.y_resptime_min)
        # y = StandardScaler().fit_transform(y.reshape((len(y),1)))
        # y = y.flatten()
        if vis == True:
            plt.subplot(121)
            n, bins, patches = plt.hist(y_resptime, 50, density=True, facecolor='g', alpha=0.75)
            plt.grid(True)
            plt.subplot(122)
            n, bins, patches = plt.hist(y_answer, 50, density=True, facecolor='g', alpha=0.75)
            plt.grid(True)
            plt.show()

        if data_aug == True:
            x, y_resptime, y_answer = dataset_augment(x, y_resptime, y_answer)

        if shuffle == True:
            indices = np.arange(len(y_resptime))
            np.random.shuffle(indices)
            x, y_resptime, y_answer = x[indices].copy(), y_resptime[indices].copy(), y_answer[indices].copy()
            data_set_array = np.array(data_set)
            data_set_array = data_set_array[indices].copy()
            data_set = pd.DataFrame(data_set_array, columns = current_head)
        # print(y_resptime[:5]*(self.y_resptime_max-self.y_resptime_min)+self.y_resptime_min)
        # print(y_answer[:5])
        # print(data_set[:5])

        split_at = int(split_rate * len(y_resptime))
        x_train, x_test = x[:split_at], x[split_at:]
        y_resptime_train, y_resptime_test = y_resptime[:split_at], y_resptime[split_at:]
        y_answer_train, y_answer_test = y_answer[:split_at], y_answer[split_at:]
        self.data_set_train, self.data_set_test = data_set[:split_at], data_set[split_at:]

        print('x_train.shape: ', x_train.shape, ', y_answer_train.shape: ', y_answer_train.shape, ', y_resptime_train.shape: ', y_resptime_train.shape)
        print('x_test.shape: ', x_test.shape, ', y_answer_test.shape: ', y_answer_test.shape, ', y_resptime_test.shape: ', y_resptime_test.shape)

        return x_train, y_answer_train, y_resptime_train, x_test, y_answer_test, y_resptime_test

    def dataset_augment(self,x,y1,y2,num=1):
        # x_aug, y_aug = x.copy(), y.copy()
        # for ni in range(num):
        #     x_noise = np.random.normal(0,0.001,x_aug.shape)
        #     print('np.max(x_noise): ', np.max(x_noise))
        #     x_add = x_aug.copy() + x_noise
        #     y_add = y_aug.copy() + y_aug.copy()
        #     x_aug, y_aug = np.concatenate((x_aug,x_add),axis=0), np.concatenate((y_aug,y_add),axis=0)

        x_noise = np.random.normal(0,0.001,x.shape)
        print('np.max(x_noise): ', np.max(x_noise))
        x_aug, y_aug1, y_aug2 = np.concatenate((x,x_noise),axis=0), np.concatenate((y1,y1),axis=0), np.concatenate((y2,y2),axis=0)

        return x_aug, y_aug1, y_aug2
    
    def model_init(self):
        self.model_answer = svm.SVC(probability = True)
        self.model_resptime = svm.SVR()
        return self.model_answer, self.model_resptime

    def model_train(self, x_train, y_answer_train, y_resptime_train, save_model_path, save_model = False):
        '''save_model_path = ['model_answer path','model_resptime path']'''
        self.model_answer.fit(x_train, y_answer_train)
        self.model_resptime.fit(x_train, y_resptime_train)
        if save_model == True:
            dump(self.model_answer, save_model_path[0])
            dump(self.model_resptime, save_model_path[1])
            dump(self.scaler, save_model_path[1].split('.')[0]+'_scaler.joblib')
            dump(self.y_resptime_max, save_model_path[1].split('.')[0]+'_ymax.joblib')
            dump(self.y_resptime_min, save_model_path[1].split('.')[0]+'_ymin.joblib')

        return self.model_answer, self.model_resptime

    def model_load(self, save_model_path):
        self.model_answer = load(save_model_path[0])
        self.model_resptime = load(save_model_path[1])
        self.scaler = load(save_model_path[1].split('.')[0]+'_scaler.joblib')
        self.y_resptime_max = load(save_model_path[1].split('.')[0]+'_ymax.joblib')
        self.y_resptime_min = load(save_model_path[1].split('.')[0]+'_ymin.joblib')

    def model_pred(self, math_question_str, question_ids):
        x_math = np.zeros((len(math_question_str), self.MAXLEN, len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(math_question_str):
            x_math[i] = self.ctable.encode(sentence, self.MAXLEN)
        x_out = self.model_accuracy_exp.predict(x_math)
        x = x_out.reshape((x_out.shape[0],x_out.shape[2]))
        x = np.concatenate((x,np.array(question_ids).reshape((len(question_ids),1))),axis=1)
        x = self.scaler.transform(x)

        y_answer_predicted = self.model_answer.predict(x)
        y_answer_predicted_prob = self.model_answer.predict_proba(x)
        y_resptime_predicted = self.model_resptime.predict(x)
        y_resptime_predicted = y_resptime_predicted * (self.y_resptime_max-self.y_resptime_min) + self.y_resptime_min

        return y_answer_predicted, y_answer_predicted_prob, y_resptime_predicted


    def model_eval(self, x_test, y_answer_test, y_resptime_test, vis = True):     
        y_answer_predicted = self.model_answer.predict(x_test)
        y_resptime_predicted = self.model_resptime.predict(x_test)
        y_resptime_predicted = y_resptime_predicted * (self.y_resptime_max - self.y_resptime_min) + self.y_resptime_min
        y_resptime_test = y_resptime_test * (self.y_resptime_max - self.y_resptime_min) + self.y_resptime_min
        # print('y_predicted: ', y_predicted)
        accu_score = metrics.accuracy_score(y_answer_test, y_answer_predicted)
        f1_score = metrics.f1_score(y_answer_test, y_answer_predicted)
        precision_score = metrics.precision_score(y_answer_test, y_answer_predicted)
        recall_score = metrics.recall_score(y_answer_test, y_answer_predicted)
        mae = metrics.mean_absolute_error(y_resptime_test, y_resptime_predicted)
        mape = metrics.mean_absolute_percentage_error(y_resptime_test, y_resptime_predicted)
        # output_metric = pd.DataFrame([[accu_score, f1_score, precision_score, recall_score, mae]],columns = ['accu','f1','precision','recall','mse'])
        print(f'answer: accu: {accu_score}, f1: {f1_score}, prec: {precision_score}, recall: {recall_score}. resptime: mae: {mae}, mape: {mape}')

        if vis == True:
            plt.subplot(221)
            plt.scatter(y_resptime_predicted, y_resptime_test, edgecolors=(0, 0, 1))
            plt.plot([y_resptime_test.min(), y_resptime_test.max()], [y_resptime_test.min(), y_resptime_test.max()], 'r--', lw=3)

            plt.subplot(222)
            plt.plot(y_resptime_test, label='gt')
            plt.plot(y_resptime_predicted, label='pred')
            plt.legend()

            plt.subplot(223)
            plt.plot(y_answer_test, label='gt')
            plt.plot(y_answer_predicted, label='pred')
            plt.legend()

            plt.subplot(224)
            y_answer_predicted_proba = self.model_answer.predict_proba(x_test)
            # print('y_answer_predicted_proba: ', y_answer_predicted_proba[:,0:1])
            plt.plot(y_answer_predicted_proba[:,0:1], label='y_answer_predicted_proba')
            plt.legend()

            plt.savefig('answer_resptime_pred.png')
            plt.show()
        return [accu_score, f1_score, precision_score, recall_score, mae, mape]


def individual_user_model(withfeedback = False, math_encode = 2, add_day = False, norm = False):
    MAXLEN, chars = 11, "0123456789≡(mod) "
    model_accuracy_path = '../math_answer_agent/math_answer_model'

    prelix = ''
    if withfeedback == True:
        prelix += '_feedback'
    if math_encode == 2:
        prelix += '_encode'
    elif math_encode == 1:
        prelix += '_string'
    if add_day == True:
        prelix += '_day'
    if norm == True:
        prelix += '_norm'

    inp_train_path, inp_test_path = '../dataset/timecare~25k_train.csv', '../dataset/timecare~25k_test.csv'
    oup_train_path, oup_test_path = 'timecare~25k_estimate_train_ind_user'+prelix+'.csv', 'timecare~25k_estimate_test_ind_user'+prelix+'.csv'
    metrics_result_path = 'user_metrics'+prelix+'.csv'

    max_time = 10
    # none: [1 2 35 33 5 6 38 14 15 18 21], static: [34 37 40 9 8 13 17 22 23 30 31], random: [0 7 39 41 12 16 19 20 24 26], rule: [32 3 4 36 10 11 25 27 28 29]

    dataset_all = pd.read_csv(inp_train_path)
    groups_all = ['none','static','random','rule']
    new_head = list(dataset_all.columns.values) + ['est_answer','est_answer_proba','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['user_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape'])
    counter_i = 0
    for group_name in groups_all:
        dataset_group = dataset_all[dataset_all['group'] == group_name]
        user_group = list(set(dataset_group['user_id']))
        for user_id in user_group:
            print(f'user id: {user_id}')  
            answer_resptime_model = answer_classifier_resptime_regressor(model_accuracy_path, MAXLEN, chars)

            x_train, y_answer_train, y_resptime_train, _, _, _ = answer_resptime_model.data_prepare('user', user_id, inp_train_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 1, max_time = max_time, ex_out = False, group = group_name, vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day, norm = norm, dataset_type = 'train')
            data_set_train = answer_resptime_model.data_set_train
            y_resptime_max_train, y_resptime_min_train = answer_resptime_model.y_resptime_max, answer_resptime_model.y_resptime_min
            x_test, y_answer_test, y_resptime_test, _, _, _ = answer_resptime_model.data_prepare('user', user_id, inp_test_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 1, max_time = max_time, ex_out = False, group = group_name, vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day, norm = norm, dataset_type = 'test')
            data_set_test = answer_resptime_model.data_set_train
            y_resptime_max_test, y_resptime_min_test = answer_resptime_model.y_resptime_max, answer_resptime_model.y_resptime_min
            # x_train, y_answer_train, y_resptime_train, x_test, y_answer_test, y_resptime_test = answer_resptime_model.data_prepare('user', user_id, inp_train_path, MAXLEN, chars, model_accuracy_path, shuffle = True, split_rate = 0.8, max_time = max_time, ex_out = True, group = group_name, vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day)
            answer_resptime_model.model_init()
            save_model_path=['user_model/'+str(int(user_id))+'_accu'+prelix+'.joblib', 'user_model/'+str(int(user_id))+'_resp'+prelix+'.joblib']
            answer_resptime_model.model_train(x_train, y_answer_train, y_resptime_train, save_model_path=save_model_path, save_model=True)
            answer_resptime_model.y_resptime_max = y_resptime_max_train
            answer_resptime_model.y_resptime_min = y_resptime_min_train
            train_metric = answer_resptime_model.model_eval(x_train, y_answer_train, y_resptime_train, vis = False)
            answer_resptime_model.y_resptime_max = y_resptime_max_test
            answer_resptime_model.y_resptime_min = y_resptime_min_test
            test_metric = answer_resptime_model.model_eval(x_test, y_answer_test, y_resptime_test, vis = False)
            model_metrics_all.loc[counter_i] = [user_id] + train_metric + test_metric

            resptime_train_estimated, resptime_test_estimated = answer_resptime_model.model_resptime.predict(x_train), answer_resptime_model.model_resptime.predict(x_test)
            resptime_train_estimated = resptime_train_estimated * (y_resptime_max_train - y_resptime_min_train) + y_resptime_min_train
            resptime_test_estimated = resptime_test_estimated * (y_resptime_max_test - y_resptime_min_test) + y_resptime_min_test
            resptime_train_estimated, resptime_test_estimated = resptime_train_estimated.reshape((len(resptime_train_estimated),1)), resptime_test_estimated.reshape((len(resptime_test_estimated),1))
            
            answer_train_estimated, answer_test_estimated = answer_resptime_model.model_answer.predict(x_train), answer_resptime_model.model_answer.predict(x_test)
            answer_train_estimated, answer_test_estimated = answer_train_estimated.reshape((len(answer_train_estimated),1)), answer_test_estimated.reshape((len(answer_test_estimated),1))
            answer_proba_train_estimated, answer_proba_test_estimated = answer_resptime_model.model_answer.predict_proba(x_train), answer_resptime_model.model_answer.predict_proba(x_test)
            answer_proba_train_estimated, answer_proba_test_estimated = np.around(answer_proba_train_estimated[:,0:1],decimals=8), np.around(answer_proba_test_estimated[:,0:1],decimals=8)

            dataset_train_piece = pd.DataFrame(np.concatenate((np.array(data_set_train),answer_train_estimated, answer_proba_train_estimated,resptime_train_estimated),axis=1), columns=new_head)
            dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test),answer_test_estimated, answer_proba_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
            estimated_dataset_train = pd.concat([estimated_dataset_train, dataset_train_piece],ignore_index=True)
            estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)

            counter_i += 1

    estimated_dataset_train.to_csv(oup_train_path,index=False)
    estimated_dataset_test.to_csv(oup_test_path,index=False)
    model_metrics_all.to_csv(metrics_result_path,index=False)

def group_model(withfeedback = False, math_encode = 2, add_day = False, norm = False):
    MAXLEN, chars = 11, "0123456789≡(mod) "
    model_accuracy_path = '../math_answer_agent/math_answer_model'
    
    prelix = ''
    if withfeedback == True:
        prelix += '_feedback'
    if math_encode == 2:
        prelix += '_encode'
    elif math_encode == 1:
        prelix += '_string'
    if add_day == True:
        prelix += '_day'
    if norm == True:
        prelix += '_norm'

    inp_train_path, inp_test_path = '../dataset/timecare~25k_train.csv', '../dataset/timecare~25k_test.csv'
    oup_train_path, oup_test_path = 'timecare~25k_estimate_train_group'+prelix+'.csv', 'timecare~25k_estimate_test_group'+prelix+'.csv'
    metrics_result_path = 'group_metrics'+prelix+'.csv'
    
    max_time = 10
    # none: [1 2 35 33 5 6 38 14 15 18 21], static: [34 37 40 9 8 13 17 22 23 30 31], random: [0 7 39 41 12 16 19 20 24 26], rule: [32 3 4 36 10 11 25 27 28 29]

    train_set = pd.read_csv(inp_train_path)
    groups_all = ['none','static','random','rule']
    new_head = list(train_set.columns.values) + ['est_answer','est_answer_proba','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['group','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape'])
    counter_i = 0
    for group_name in groups_all:
        print(f'group: {group_name}')  
        answer_resptime_model = answer_classifier_resptime_regressor(model_accuracy_path, MAXLEN, chars)
        # select_type, user_id, input_dataset_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 0.8, max_time = 10, ex_out = True, group = 'none', vis = True, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = False, math_encode = True, add_day = False
        x_train, y_answer_train, y_resptime_train, _, _, _ = answer_resptime_model.data_prepare('group', -1, inp_train_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 1, max_time = max_time, ex_out = False, group = group_name, vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day, norm = norm, dataset_type = 'train')
        data_set_train = answer_resptime_model.data_set_train
        y_resptime_max_train, y_resptime_min_train = answer_resptime_model.y_resptime_max, answer_resptime_model.y_resptime_min
        x_test, y_answer_test, y_resptime_test, _, _, _ = answer_resptime_model.data_prepare('group', -1, inp_test_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 1, max_time = max_time, ex_out = False, group = group_name, vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day, norm = norm, dataset_type = 'test')
        data_set_test = answer_resptime_model.data_set_train
        y_resptime_max_test, y_resptime_min_test = answer_resptime_model.y_resptime_max, answer_resptime_model.y_resptime_min
        answer_resptime_model.model_init()
        save_model_path=['group_model/'+group_name+'_accu'+prelix+'.joblib', 'group_model/'+group_name+'_resp'+prelix+'.joblib']
        answer_resptime_model.model_train(x_train, y_answer_train, y_resptime_train, save_model_path=save_model_path, save_model=True)
        answer_resptime_model.y_resptime_max = y_resptime_max_train
        answer_resptime_model.y_resptime_min = y_resptime_min_train
        train_metric = answer_resptime_model.model_eval(x_train, y_answer_train, y_resptime_train, vis = False)
        answer_resptime_model.y_resptime_max = y_resptime_max_test
        answer_resptime_model.y_resptime_min = y_resptime_min_test
        test_metric = answer_resptime_model.model_eval(x_test, y_answer_test, y_resptime_test, vis = False)
        model_metrics_all.loc[counter_i] = [group_name] + train_metric + test_metric

        resptime_train_estimated, resptime_test_estimated = answer_resptime_model.model_resptime.predict(x_train), answer_resptime_model.model_resptime.predict(x_test)
        resptime_train_estimated = resptime_train_estimated * (y_resptime_max_train - y_resptime_min_train) + y_resptime_min_train
        resptime_test_estimated = resptime_test_estimated * (y_resptime_max_test - y_resptime_min_test) + y_resptime_min_test
        resptime_train_estimated, resptime_test_estimated = resptime_train_estimated.reshape((len(resptime_train_estimated),1)), resptime_test_estimated.reshape((len(resptime_test_estimated),1))
            
        answer_train_estimated, answer_test_estimated = answer_resptime_model.model_answer.predict(x_train), answer_resptime_model.model_answer.predict(x_test)
        answer_train_estimated, answer_test_estimated = answer_train_estimated.reshape((len(answer_train_estimated),1)), answer_test_estimated.reshape((len(answer_test_estimated),1))
        answer_proba_train_estimated, answer_proba_test_estimated = answer_resptime_model.model_answer.predict_proba(x_train), answer_resptime_model.model_answer.predict_proba(x_test)
        answer_proba_train_estimated, answer_proba_test_estimated = np.around(answer_proba_train_estimated[:,0:1],decimals=8), np.around(answer_proba_test_estimated[:,0:1],decimals=8)

        dataset_train_piece = pd.DataFrame(np.concatenate((np.array(data_set_train),answer_train_estimated, answer_proba_train_estimated,resptime_train_estimated),axis=1), columns=new_head)
        dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test),answer_test_estimated, answer_proba_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
        estimated_dataset_train = pd.concat([estimated_dataset_train, dataset_train_piece],ignore_index=True)
        estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)

        counter_i += 1

    estimated_dataset_train.to_csv(oup_train_path,index=False)
    estimated_dataset_test.to_csv(oup_test_path,index=False)
    model_metrics_all.to_csv(metrics_result_path,index=False)

def general_model(withfeedback = False, math_encode = 2, add_day = False, norm = False):
    MAXLEN, chars = 11, "0123456789≡(mod) "
    model_accuracy_path = '../math_answer_agent/math_answer_model'

    prelix = ''
    if withfeedback == True:
        prelix += '_feedback'
    if math_encode == 2:
        prelix += '_encode'
    elif math_encode == 1:
        prelix += '_string'
    if add_day == True:
        prelix += '_day'
    if norm == True:
        prelix += '_norm'

    inp_train_path, inp_test_path = '../dataset/timecare~25k_train.csv', '../dataset/timecare~25k_test.csv'
    oup_train_path, oup_test_path = 'timecare~25k_estimate_train_general'+prelix+'.csv', 'timecare~25k_estimate_test_general'+prelix+'.csv'
    metrics_result_path = 'general_metrics'+prelix+'.csv'

    max_time = 10
    # none: [1 2 35 33 5 6 38 14 15 18 21], static: [34 37 40 9 8 13 17 22 23 30 31], random: [0 7 39 41 12 16 19 20 24 26], rule: [32 3 4 36 10 11 25 27 28 29]

    train_set = pd.read_csv(inp_train_path)
    groups_all = ['none','static','random','rule']
    new_head = list(train_set.columns.values) + ['est_answer','est_answer_proba','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['general','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape'])
    counter_i = 0
        
    answer_resptime_model = answer_classifier_resptime_regressor(model_accuracy_path, MAXLEN, chars)
    # select_type, user_id, input_dataset_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 0.8, max_time = 10, ex_out = True, group = 'none', vis = True, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = False, math_encode = True, add_day = False
    x_train, y_answer_train, y_resptime_train, _, _, _ = answer_resptime_model.data_prepare('all', -1, inp_train_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 1, max_time = max_time, ex_out = False, group = 'all', vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day, norm = norm, dataset_type = 'train')
    data_set_train = answer_resptime_model.data_set_train
    y_resptime_max_train, y_resptime_min_train = answer_resptime_model.y_resptime_max, answer_resptime_model.y_resptime_min
    x_test, y_answer_test, y_resptime_test, _, _, _ = answer_resptime_model.data_prepare('all', -1, inp_test_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 1, max_time = max_time, ex_out = False, group = 'all', vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day, norm = norm, dataset_type = 'test')
    data_set_test = answer_resptime_model.data_set_train
    y_resptime_max_test, y_resptime_min_test = answer_resptime_model.y_resptime_max, answer_resptime_model.y_resptime_min
    answer_resptime_model.model_init()
    save_model_path=['general_model/'+'_accu'+prelix+'.joblib', 'general_model/'+'_resp'+prelix+'.joblib']
    answer_resptime_model.model_train(x_train, y_answer_train, y_resptime_train, save_model_path=save_model_path, save_model=True)
    answer_resptime_model.y_resptime_max = y_resptime_max_train
    answer_resptime_model.y_resptime_min = y_resptime_min_train
    train_metric = answer_resptime_model.model_eval(x_train, y_answer_train, y_resptime_train, vis = False)
    answer_resptime_model.y_resptime_max = y_resptime_max_test
    answer_resptime_model.y_resptime_min = y_resptime_min_test
    test_metric = answer_resptime_model.model_eval(x_test, y_answer_test, y_resptime_test, vis = False)
    model_metrics_all.loc[counter_i] = ['general'] + train_metric + test_metric

    resptime_train_estimated, resptime_test_estimated = answer_resptime_model.model_resptime.predict(x_train), answer_resptime_model.model_resptime.predict(x_test)
    resptime_train_estimated = resptime_train_estimated * (y_resptime_max_train - y_resptime_min_train) + y_resptime_min_train
    resptime_test_estimated = resptime_test_estimated * (y_resptime_max_test - y_resptime_min_test) + y_resptime_min_test
    resptime_train_estimated, resptime_test_estimated = resptime_train_estimated.reshape((len(resptime_train_estimated),1)), resptime_test_estimated.reshape((len(resptime_test_estimated),1))
            
    answer_train_estimated, answer_test_estimated = answer_resptime_model.model_answer.predict(x_train), answer_resptime_model.model_answer.predict(x_test)
    answer_train_estimated, answer_test_estimated = answer_train_estimated.reshape((len(answer_train_estimated),1)), answer_test_estimated.reshape((len(answer_test_estimated),1))
    answer_proba_train_estimated, answer_proba_test_estimated = answer_resptime_model.model_answer.predict_proba(x_train), answer_resptime_model.model_answer.predict_proba(x_test)
    answer_proba_train_estimated, answer_proba_test_estimated = np.around(answer_proba_train_estimated[:,0:1],decimals=8), np.around(answer_proba_test_estimated[:,0:1],decimals=8)

    dataset_train_piece = pd.DataFrame(np.concatenate((np.array(data_set_train),answer_train_estimated, answer_proba_train_estimated,resptime_train_estimated),axis=1), columns=new_head)
    dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test),answer_test_estimated, answer_proba_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
    estimated_dataset_train = pd.concat([estimated_dataset_train, dataset_train_piece],ignore_index=True)
    estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)

    counter_i += 1

    estimated_dataset_train.to_csv(oup_train_path,index=False)
    estimated_dataset_test.to_csv(oup_test_path,index=False)
    model_metrics_all.to_csv(metrics_result_path,index=False)

def LOPO_model(withfeedback = False, math_encode = 2, add_day = False, norm = False):
    MAXLEN, chars = 11, "0123456789≡(mod) "
    model_accuracy_path = '../math_answer_agent/math_answer_model'

    prelix = ''
    if withfeedback == True:
        prelix += '_feedback'
    if math_encode == 2:
        prelix += '_encode'
    elif math_encode == 1:
        prelix += '_string'
    if add_day == True:
        prelix += '_day'
    if norm == True:
        prelix += '_norm'

    inp_path = '../dataset/timecare~25k_raw_filtered_invalid_user.csv'
    oup_train_path, oup_test_path = 'timecare~25k_estimate_train_lopo'+prelix+'.csv', 'timecare~25k_estimate_test_lopo'+prelix+'.csv'
    metrics_result_path = 'lopo_metrics'+prelix+'.csv'

    max_time = 10

    dataset_all = pd.read_csv(inp_path)
    groups_all = ['none','static','random','rule']
    new_head = list(dataset_all.columns.values) + ['est_answer','est_answer_proba','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['user_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape'])
    counter_i = 0
    if os.path.exists('lopo_model/') == False:
        os.mkdir('lopo_model/')
    for group_name in groups_all:
        if os.path.exists('lopo_model/'+group_name) == False:
            os.mkdir('lopo_model/'+group_name)
        dataset_group = dataset_all[dataset_all['group'] == group_name]
        user_group = list(set(dataset_group['user_id']))
        user_group.sort()
        for user_id in user_group:
            if os.path.exists('lopo_model/'+group_name+'/'+str(int(user_id))) == False:
                os.mkdir('lopo_model/'+group_name+'/'+str(int(user_id)))
            print(f'user id: {user_id}')  

            user_group_train = user_group.copy()
            del user_group_train[user_group_train.index(user_id)]

            answer_resptime_model = answer_classifier_resptime_regressor(model_accuracy_path, MAXLEN, chars)

            x_train, y_answer_train, y_resptime_train, _, _, _ = answer_resptime_model.data_prepare('user_list', None, inp_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 1, max_time = max_time, ex_out = False, group = None, vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day, norm = norm, dataset_type = 'train', user_list=user_group_train)
            data_set_train = answer_resptime_model.data_set_train
            y_resptime_max_train, y_resptime_min_train = answer_resptime_model.y_resptime_max, answer_resptime_model.y_resptime_min
            x_test, y_answer_test, y_resptime_test, _, _, _ = answer_resptime_model.data_prepare('user', user_id, inp_path, MAXLEN, chars, model_accuracy_path, shuffle = False, split_rate = 1, max_time = max_time, ex_out = False, group = None, vis = False, y_scale_resptime = True, data_aug = False, pca_flag = False, pca_num = 10, withfeedback = withfeedback, math_encode=math_encode, add_day = add_day, norm = norm, dataset_type = 'test', user_list=None)
            data_set_test = answer_resptime_model.data_set_train
            y_resptime_max_test, y_resptime_min_test = answer_resptime_model.y_resptime_max, answer_resptime_model.y_resptime_min
            answer_resptime_model.model_init()
            save_model_path=['lopo_model/'+group_name+'/'+str(int(user_id))+'/'+str(int(user_id))+'_accu'+prelix+'.joblib', 'lopo_model/'+group_name+'/'+str(int(user_id))+'/'+str(int(user_id))+'_resp'+prelix+'.joblib']
            answer_resptime_model.model_train(x_train, y_answer_train, y_resptime_train, save_model_path=save_model_path, save_model=True)
            answer_resptime_model.y_resptime_max = y_resptime_max_train
            answer_resptime_model.y_resptime_min = y_resptime_min_train
            train_metric = answer_resptime_model.model_eval(x_train, y_answer_train, y_resptime_train, vis = False)
            answer_resptime_model.y_resptime_max = y_resptime_max_test
            answer_resptime_model.y_resptime_min = y_resptime_min_test
            test_metric = answer_resptime_model.model_eval(x_test, y_answer_test, y_resptime_test, vis = False)
            model_metrics_all.loc[counter_i] = [user_id] + train_metric + test_metric

            resptime_train_estimated, resptime_test_estimated = answer_resptime_model.model_resptime.predict(x_train), answer_resptime_model.model_resptime.predict(x_test)
            resptime_train_estimated = resptime_train_estimated * (y_resptime_max_train - y_resptime_min_train) + y_resptime_min_train
            resptime_test_estimated = resptime_test_estimated * (y_resptime_max_test - y_resptime_min_test) + y_resptime_min_test
            resptime_train_estimated, resptime_test_estimated = resptime_train_estimated.reshape((len(resptime_train_estimated),1)), resptime_test_estimated.reshape((len(resptime_test_estimated),1))
            
            answer_train_estimated, answer_test_estimated = answer_resptime_model.model_answer.predict(x_train), answer_resptime_model.model_answer.predict(x_test)
            answer_train_estimated, answer_test_estimated = answer_train_estimated.reshape((len(answer_train_estimated),1)), answer_test_estimated.reshape((len(answer_test_estimated),1))
            answer_proba_train_estimated, answer_proba_test_estimated = answer_resptime_model.model_answer.predict_proba(x_train), answer_resptime_model.model_answer.predict_proba(x_test)
            answer_proba_train_estimated, answer_proba_test_estimated = np.around(answer_proba_train_estimated[:,0:1],decimals=8), np.around(answer_proba_test_estimated[:,0:1],decimals=8)

            dataset_train_piece = pd.DataFrame(np.concatenate((np.array(data_set_train),answer_train_estimated, answer_proba_train_estimated,resptime_train_estimated),axis=1), columns=new_head)
            dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test),answer_test_estimated, answer_proba_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
            estimated_dataset_train = pd.concat([estimated_dataset_train, dataset_train_piece],ignore_index=True)
            estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)

            dataset_train_piece.to_csv('lopo_model/'+group_name+'/'+str(int(user_id))+'/trainset.csv',index=False)
            dataset_test_piece.to_csv('lopo_model/'+group_name+'/'+str(int(user_id))+'/testset.csv',index=False)

            counter_i += 1

    estimated_dataset_train.to_csv(oup_train_path,index=False)
    estimated_dataset_test.to_csv(oup_test_path,index=False)
    model_metrics_all.to_csv(metrics_result_path,index=False)

# math_encode: 0-numeric math-''. 1-encode string-'string', 2-features from math answer agent-'encode'

# individual_user_model(withfeedback = False, math_encode = 2, add_day = False, norm = True)
# group_model(withfeedback = False, math_encode = 2, add_day = False, norm = True)
# # general_model(withfeedback = False, math_encode = 2, add_day = False, norm = True)
# LOPO_model(withfeedback = False, math_encode = 2, add_day = False, norm = True)

# individual_user_model(withfeedback = True, math_encode = 2, add_day = False, norm = True)
# group_model(withfeedback = True, math_encode = 2, add_day = False, norm = True)
# general_model(withfeedback = True, math_encode = 2, add_day = False, norm = True)

# individual_user_model(withfeedback = False, math_encode = 1, add_day = False, norm = True)
# group_model(withfeedback = False, math_encode = 1, add_day = False, norm = True)
# general_model(withfeedback = False, math_encode = 1, add_day = False, norm = True)


# individual_user_model(withfeedback = True, math_encode = 1, add_day = False, norm = True)
# group_model(withfeedback = True, math_encode = 1, add_day = False, norm = True)
# general_model(withfeedback = True, math_encode = 1, add_day = False, norm = True)



# individual_user_model(withfeedback = False, math_encode = 0, add_day = False, norm = True)
# group_model(withfeedback = False, math_encode = 0, add_day = False, norm = True)
# general_model(withfeedback = False, math_encode = 0, add_day = False, norm = True)

# individual_user_model(withfeedback = True, math_encode = 0, add_day = False, norm = True)
# group_model(withfeedback = True, math_encode = 0, add_day = False, norm = True)
# general_model(withfeedback = True, math_encode = 0, add_day = False, norm = True)
