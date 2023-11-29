import gym
from gym import spaces
import cv2
from PIL import Image
import os, sys, math, time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(4)

import pickle
from joblib import dump, load
import numpy as np
from numpy.random import seed
seed(4)
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats
from math import sqrt
from scipy.stats import norm

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

def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out

def wiener_curve(x_start, x_end, y_start, y_end, frequency = 10, delta = 0.01):
    step_num = int(abs(x_end-x_start)*frequency)
    dt = abs(y_end-y_start)/step_num
    noise_array = brownian(0, step_num, dt, delta)
    return noise_array

def sigmoid_curve(x_start, x_end, y_start, y_end, frequency = 10, init_interval = 20):
    assert x_end >= x_start and y_end >= y_start
    step_num = int(abs(x_end-x_start)*frequency)
    x_init_start, x_init_end = -init_interval/2, init_interval/2
    x = np.linspace(x_init_start, x_init_end, step_num)
    y = 1/(1 + np.exp(-x))
    y_init_start, y_init_end = np.min(y), np.max(y)
    x = x * (abs(x_end-x_start)/abs(x_init_end-x_init_start)) 
    x_current_start, x_current_end = np.min(x), np.max(x)
    x = x + x_start - x_current_start 
    y_init_end = np.around(y_init_end, decimals = 4)
    y_init_start = np.around(y_init_start, decimals = 4)
    # print('y_init_end, y_init_start: ', y_init_end, y_init_start)
    print(y_init_end)
    y = y * (abs(y_end-y_start)/abs(y_init_end-y_init_start)) + y_start - y_init_start
    return x, y


class MathModularEnv(gym.Env):
    def __init__(self, config):
        super(MathModularEnv, self).__init__()
        self.max_time_set, self.refresh_frequency = config['max_time_set'], config['refresh_frequency']
        self.noise_delta, self.stimuli_discounter = config['noise_delta'], config['stimuli_discounter']
        self.chars, self.MaxMathLen = config['chars'], config['MaxMathLen']
        self.encode = config['encode']
        self.ctable = CharacterTable(self.chars)
        self.fine_level = config['fine_level']
        
        n_actions = 3 # 0: -1, 1: 0, 2: +1.
        if config['fine_level'] == 'discrete':
            self.action_space = spaces.Discrete(n_actions)
        else:
            self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float64)
        assert config['encode'] != None
        if config['encode'] == True:
            self.observation_space = spaces.Dict(
                spaces={
                    "math": gym.spaces.Box(0, 2, [self.MaxMathLen, len(self.chars)], dtype=np.float64), 
                    "feedback_img": gym.spaces.Box(0, 255, [3,100,1240], dtype=np.uint8),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                spaces={
                    "math": gym.spaces.Box(0, 100, [1, 3], dtype=np.float64), 
                    "feedback_img": gym.spaces.Box(0, 255, [3,100,1240], dtype=np.uint8),
                }
            )

        self.dataset = pd.read_csv(config['dataset_all_path'])
        # self.dataset = pd.DataFrame(np.array(self.dataset)[:10],columns=self.dataset.columns.values)
        assert config['select_type'] == 'all' or config['select_type'] == 'group' or config['select_type'] == 'user'
        if config['select_type'] == 'all':
            dataset_filtered = np.array(self.dataset[(self.dataset['resptime'] < self.max_time_set)])
        elif config['select_type'] == 'group':
            dataset_filtered = np.array(self.dataset[(self.dataset['group']==config['group']) & (self.dataset['resptime'] < self.max_time_set)])
        else:
            dataset_filtered = np.array(self.dataset[(self.dataset['user_id']==config['user_id']) & (self.dataset['resptime'] < self.max_time_set)])
        self.dataset = pd.DataFrame(dataset_filtered,columns=self.dataset.columns.values)

        self.dataset_len = self.dataset.shape[0]
        print('self.dataset_len: ', self.dataset_len, ', true num: ', len(self.dataset[self.dataset['answer']==1]))
        self.feedback_dataset_path = config['feedback_dataset_path']
        
        self.episode_counter, self.step_counter, self.accumulation = 0, 0, 0
        self.math_question, self.current_feedback, self.user_answer, self.user_resptime = None, None, None, None
        self.modular_math_generator = modular_math_generator()
        self.state_dict = {'math': None, 'feedback_img': None}
        self.agent_action, self.agent_answer, self.agent_resptime = 0, None, None
        self.agent_action_set = []
        self.env_type = config['env_type']
        self.test_format = config['test_format']
        assert self.env_type == 'train' or self.env_type == 'test'
        print('self.test_format: ', self.test_format)
        assert self.test_format == 'all' or self.test_format == 'random' or self.test_format == None
        

    def generate_feedback_img(self, feedback_flag, current_step_num):
        assert feedback_flag == 1 or feedback_flag == 0
        feedback_without_path = self.feedback_dataset_path+'/frame0.jpg'
        feedback_with_path = self.feedback_dataset_path+'/frame'+str(int(60/self.refresh_frequency)*(current_step_num%int(5*self.refresh_frequency)))+'.jpg' 
        feedback_img_path = feedback_without_path if feedback_flag == 0 else feedback_with_path
       
        feedback_image = Image.open(feedback_img_path)
        feedback_image_arr = np.array(feedback_image)[300:400,20:1260]
        feedback_image_arr = np.moveaxis(feedback_image_arr, -1, 0)

        return feedback_image_arr

    def generate_math_sample(self):
        # user_id,group,day,session,question_id,num_1,num_2,num_3,feedback,accuracy,resptime,focus,focustime,anxiety,anxietytime
        if self.env_type == 'train': 
            sample_index = np.random.randint(0,self.dataset_len)
        elif self.env_type == 'test' and self.test_format == 'random':
            sample_index = np.random.randint(0,self.dataset_len)
        elif self.env_type == 'test' and self.test_format == 'all':
            sample_index = int(self.episode_counter-1)
            if sample_index == self.dataset_len:
                sample_index -= 1
                print('no need to reset or select data')
            
        print('sample_index: ', sample_index, ', self.dataset_len: ', self.dataset_len)
        each_sample = self.dataset.loc[sample_index]
        question_id, self.num_1, self.num_2, self.num_3 = each_sample['question_id'], each_sample['num_1'], each_sample['num_2'], each_sample['num_3']
        feedback, user_answer, user_resptime = each_sample['feedback'], each_sample['answer'], each_sample['resptime']
        est_answer_proba, est_resptime = each_sample['est_answer_proba'], each_sample['est_resptime']
        
        if self.encode == True:
            math_question = self.ctable.encode(str(int(self.num_1)) + '≡' + str(int(self.num_2)) + '(mod' + str(int(self.num_3)) + ')', self.MaxMathLen)
        else:
            math_question = [int(self.num_1), int(self.num_2), int(self.num_3)]
        self.math_truth = self.modular_math_generator.get_truth(self.num_1, self.num_2, self.num_3)

        return math_question, feedback, question_id, user_answer, user_resptime, est_answer_proba, est_resptime

    def reset(self):
        if self.env_type == 'test':
            print('reset env')
        self.step_counter = 0    # must before than self.generate_sample(), because self.generate_sample() will also use updated self.step_counter
        self.accumulation = 0
        self.agent_action_set = []
        self.episode_counter += 1
        self.state_dict['math'], self.current_feedback, question_id, self.user_answer, self.user_resptime, self.est_answer_proba, self.est_resptime = self.generate_math_sample()
        self.trajectory, self.est_answer, self.delta_proba = self.generate_trajectory(self.est_resptime, self.est_answer_proba, self.refresh_frequency, init_interval=20, delta = self.noise_delta)
        self.agent_resptime, self.agent_answer = None, None

        self.state_dict['feedback_img'] = self.generate_feedback_img(self.current_feedback, self.step_counter)

        return self.state_dict

    def get_reward_accu(self, terminal, agent_answer, user_answer):
        reward_accu = 0
        if (terminal == True and self.step_counter < int(self.max_time_set*self.refresh_frequency)):
            if self.est_answer != user_answer and agent_answer == user_answer:
                reward_accu = 1
            elif agent_answer != user_answer:
                reward_accu = -1
        # if terminal == True and self.step_counter >= int(self.max_time_set*self.refresh_frequency):
        #     reward_accu = -1
        return reward_accu
    
    def get_reward_resp(self, terminal, agent_resptime, user_resptime):
        # give reward only if the new agent response time is closer to real response time compared with estimated response time from SVM regressor 
        # or the agent answer is more accurate than estimated answer.
        if terminal == False or self.step_counter >= int(self.max_time_set*self.refresh_frequency):
            reward_resp = 0
        else:
            est_error_rate = abs(self.est_resptime-user_resptime)/user_resptime
            agent_error_rate = abs(agent_resptime-user_resptime)/user_resptime
            if agent_error_rate >= est_error_rate:
                reward_resp = 0
            else:
                if est_error_rate != 0:
                    improve_error_rate = abs(est_error_rate-agent_error_rate)/est_error_rate
                    reward_resp = improve_error_rate   
                else:
                    reward_resp = 0
           
        return reward_resp
    
    def get_penalty(self, terminal):
        if (terminal == True and self.step_counter >= int(self.max_time_set*self.refresh_frequency)):
            penalty = -1
        else:
            penalty = 0
        return penalty

    def get_reward(self, terminal, agent_answer, agent_resptime, user_answer, user_resptime):
        reward_accu = self.get_reward_accu(terminal, agent_answer, user_answer)
        reward_resp = self.get_reward_resp(terminal, agent_resptime, user_resptime)
        penalty = self.get_penalty(terminal)
        # reward = 0.5 * reward_accu + 0.5 * reward_resp + penalty
        # reward = reward_accu + penalty
        reward = reward_resp + penalty
        return reward
    
    def terminal_flag(self, action):
        '''return 1: finished, return 0: not finished, return -1: fail and reset'''
        if self.fine_level == 'discrete':
            assert action == 0 or action == 1 or action == 2
            assert self.est_answer == 0 or self.est_answer == 1
            if action == 0:
                self.accumulation -= self.delta_proba
            elif action == 1:
                self.accumulation += 0
            elif action == 2:
                self.accumulation += self.delta_proba
        else:
            self.accumulation += (action * self.delta_proba)
        
        if self.step_counter >= int(self.max_time_set*self.refresh_frequency):
            self.agent_answer = None
            self.agent_resptime = None
            return -1
        else:
            final_evidence = self.accumulation * self.stimuli_discounter + self.trajectory[self.step_counter]
            if self.est_answer_proba > 0.5:
                if final_evidence >= self.est_answer_proba:
                    self.agent_answer = self.est_answer
                    self.agent_resptime = self.step_counter/self.refresh_frequency
                    return 1
                # elif final_evidence <= (1-self.est_answer_proba):
                #     self.agent_answer = 1 - self.est_answer
                #     self.agent_resptime = self.step_counter/self.refresh_frequency
                #     return 1 
                else:
                    return 0
            else:
                if final_evidence >= 1-self.est_answer_proba:
                    self.agent_answer = self.est_answer
                    self.agent_resptime = self.step_counter/self.refresh_frequency
                    return 1
                # elif final_evidence <= self.est_answer_proba:
                #     self.agent_answer = 1 - self.est_answer
                #     self.agent_resptime = self.step_counter/self.refresh_frequency
                #     return 1 
                else:
                    return 0


    def generate_trajectory(self, resptime, answer_proba, frequency = 10, init_interval = 20, delta = 0.01):
        x_start, x_end = 0, resptime
        y_start = 0.5
        if answer_proba >= 0.5:
            y_end = answer_proba
            answer = 0
        else:
            y_end = 1 - answer_proba
            answer = 1
        delta_proba = abs(y_end - y_start)/(abs(x_end - x_start)*frequency)
        x, y = sigmoid_curve(x_start, x_end, y_start, y_end, frequency, init_interval)
        noise_array = wiener_curve(x_start, x_end, y_start, y_end, frequency, delta)
        final_signal = y + noise_array
        final_signal[0] = y[0]
        final_signal[-1] = y[-1]
        for fi in range(len(final_signal)-1):
            if final_signal[fi] >= final_signal[-1]:
                final_signal[fi] = final_signal[-1] - 0.0001
            elif final_signal[fi] < final_signal[0]:
                final_signal[fi] = final_signal[0]
        if len(final_signal) < int(self.max_time_set*self.refresh_frequency):
            trajectory = np.zeros((int(self.max_time_set*self.refresh_frequency))) + final_signal[-1]
            trajectory[:len(final_signal)] = final_signal
        else:
            trajectory = final_signal

        return trajectory, answer, delta_proba

    def step(self, action):
        
        self.agent_action_set.append(action)
        self.step_counter += 1
        self.agent_action = action
        
        self.state_dict['feedback_img'] = self.generate_feedback_img(self.current_feedback, self.step_counter)
        
        terminal_flag_set = self.terminal_flag(action)
        fail_flag = 1 if (terminal_flag_set == -1) else 0
        done = True if (terminal_flag_set == 1 or terminal_flag_set == -1) else False
        
        reward = self.get_reward(done, self.agent_answer, self.agent_resptime, self.user_answer, self.user_resptime)

        self.agent_accuracy = 1 if (self.agent_answer == self.math_truth) else 0

        info = {'result': str(self.agent_answer)+','+str(self.user_answer)+','+str(self.agent_resptime)+','+str(self.user_resptime)+','+str(fail_flag)+','+str(self.agent_accuracy)+','+str(self.est_answer)+','+str(self.est_resptime)}

        math_question_str = str(self.num_1) + ',' + str(self.num_2) + ',' + str(self.num_3)
        agent_selection = 'no select' if self.agent_answer == None else 'true' if self.agent_answer == 1 else 'false'
        user_selection = '---' if agent_selection == 'no select' else 'true' if self.user_answer == 1 else 'false'
        estimate_selection = '---' if agent_selection == 'no select' else 'true' if self.est_answer == 1 else 'false'
        # if self.env_type == 'train' and reward != 0:
        #     print(f'episode: {self.episode_counter}, step: {self.step_counter}, reward: {reward}, math question: {math_question_str}, agent selection (action): {agent_selection}, user selection: {user_selection}, agent resptime: {self.agent_resptime}, user resptime: {self.user_resptime}')
        if self.env_type == 'test' and done == True:
            print(f'epi: {self.episode_counter}, step: {self.step_counter}, reward: {reward}, math: {math_question_str}, agent accuray: {self.agent_accuracy}, agent answer: {agent_selection}, est answer: {estimate_selection}, est proba: {self.est_answer_proba}, user answer: {user_selection}, agent resptime: {self.agent_resptime}, estimate resptime: {self.est_resptime}, user resptime: {self.user_resptime}')
            print(f'self.agent_action_set: {self.agent_action_set}')
        return self.state_dict, reward, done, info

