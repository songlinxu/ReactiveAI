import torch as th
from stable_baselines3 import PPO, SAC, DQN, A2C
from stable_baselines3.common.envs import SimpleMultiObsEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from typing import Callable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(4)

import pickle
from joblib import dump, load
from matplotlib.cbook import boxplot_stats

from MathModularWhole import MathModularEnvWhole
from utils import _get_epi_num, _get_group_id, _get_user_data, _get_user_id, _get_group_users

import argparse, os, sys, time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--type', '-t', type=str, required=True)
args = parser.parse_args()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    train_curve_data = pd.DataFrame(np.concatenate((x.reshape((len(x),1)),y.reshape((len(y),1))),axis=1),columns=['x','y'])
    train_curve_data.to_csv(log_folder+'/train_curve.csv', index=False)
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    # plt.clf()
    # plt.close()
    fig = plt.figure()
    plt.subplot(111)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(log_folder+'/return.png')
    # plt.show()

def train(config):
    log_dir = config['log_dir']
    if os.path.exists(log_dir) == False:
        os.mkdir(log_dir)
    env_train = MathModularEnvWhole(config)
    env_train = Monitor(env_train, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    model = PPO("MultiInputPolicy",env_train,verbose=1, tensorboard_log=log_dir, seed=4)  
    timer_start = time.time()
    model.learn(total_timesteps=config['train_step'], callback=callback)
    # model.learn(total_timesteps=train_step, callback=callback, reset_num_timesteps=True)
    timer_end = time.time()
    # model.save(MODEL_PATH)

    print(f'Finished training. Training time: {timer_end-timer_start}.')
    plot_results(log_dir)
    # results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "TD3 LunarLander")

# ================================================================================================================
def save_list(input_list, output_file_path):
    with open(output_file_path,'a+') as f:
        for line_indx, line in enumerate(input_list):
            for letter_index, letter in enumerate(line):
                if letter_index == (len(line) - 1):
                    f.write(str(letter)+'\n')
                else:
                    f.write(str(letter)+',')
 
def test(config):
    env_test = MathModularEnvWhole(config)
    env_test = make_vec_env(lambda: env_test, n_envs=1)
    Test_MODEL_PATH = config['log_dir']

    print('log model path: ', Test_MODEL_PATH)
    model = PPO.load(Test_MODEL_PATH+'/best_model.zip', env=env_test)

    obs = env_test.reset()
    trial_count = 0

    agent_answer, user_answer, agent_resptime, user_resptime, est_answer, est_resptime, fail_rate = [], [], [], [], [], [], []
    while trial_count < config['test_epi_num']:
        old_obs = obs
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env_test.step(action)

        # if dones[0] == True:
        # print(f'trial count: {trial_count}')
        trial_count += 1
            # obs = env_test.reset()
            # print(info)
        info_useful = info[0]['result'].split(',')
        # print('rewards: ',rewards)
        fail_flag = float(info_useful[4])
        fail_rate.append(fail_flag)
        if fail_flag == 0:
            agent_answer.append(float(info_useful[0]))
            agent_resptime.append(float(info_useful[2]))
        else:
            agent_answer.append(-1)
            temp = float(info_useful[2])
            if temp <= 0: 
                temp_agent_resptime = 0
            elif temp >= 10:
                temp_agent_resptime = 10
            else:
                temp_agent_resptime = temp
            agent_resptime.append(temp_agent_resptime)
        user_answer.append(float(info_useful[1]))
        user_resptime.append(float(info_useful[3]))
        est_answer.append(float(info_useful[6]))
        est_resptime.append(float(info_useful[7]))
    agent_answer = np.array(agent_answer).reshape((config['test_epi_num'],1))
    user_answer = np.array(user_answer).reshape((config['test_epi_num'],1))
    agent_resptime = np.array(agent_resptime).reshape((config['test_epi_num'],1))
    user_resptime = np.array(user_resptime).reshape((config['test_epi_num'],1))
    est_answer = np.array(est_answer).reshape((config['test_epi_num'],1))
    est_resptime = np.array(est_resptime).reshape((config['test_epi_num'],1))
    fail_rate = np.array(fail_rate).reshape((config['test_epi_num'],1))

    # save all predicted results into file
    if config['save_eval_file'] == True:
        eval_data_head = ['user_answer', 'user_resptime', 'est_answer', 'est_resptime', 'agent_answer', 'agent_resptime', 'fail_rate']
        eval_data = pd.DataFrame(np.concatenate((user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate),axis=1),columns = eval_data_head)
        eval_data.to_csv(Test_MODEL_PATH+'/'+config['test_result_name']+'.csv', index=False)

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(agent_resptime, label = 'agent resptime')
    plt.plot(est_resptime, label = 'est resptime')
    plt.plot(user_resptime, label = 'user resptime')
    plt.legend()
    plt.subplot(212)
    plt.plot(agent_answer, label = 'agent answer')
    plt.plot(est_answer, label = 'est answer')
    plt.plot(user_answer, label = 'user answer')
    plt.legend()
    plt.savefig(Test_MODEL_PATH+'/'+config['test_result_name']+'.png')
    if config['vis_eval'] == True:
        plt.show()

    return user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate

def eval_metrics(y_answer_test, y_answer_predicted, y_resptime_test, y_resptime_predicted):
    init_len = len(y_answer_test)
    y_answer_test = y_answer_test.flatten().copy()
    y_answer_predicted = y_answer_predicted.flatten().copy()
    y_resptime_test = y_resptime_test.flatten().copy()
    y_resptime_predicted = y_resptime_predicted.flatten().copy()
    
    useful_index = (y_answer_predicted != -1)
    y_answer_test = y_answer_test[useful_index]
    y_answer_predicted = y_answer_predicted[useful_index]
    y_resptime_test = y_resptime_test[useful_index]
    y_resptime_predicted = y_resptime_predicted[useful_index]

    fail_rate = 1-float(len(y_resptime_predicted)/init_len)
    accu_score = metrics.accuracy_score(y_answer_test, y_answer_predicted)
    f1_score = metrics.f1_score(y_answer_test, y_answer_predicted)
    precision_score = metrics.precision_score(y_answer_test, y_answer_predicted)
    recall_score = metrics.recall_score(y_answer_test, y_answer_predicted)
    mae = metrics.mean_absolute_error(y_resptime_test, y_resptime_predicted)
    mape = metrics.mean_absolute_percentage_error(y_resptime_test, y_resptime_predicted)
    print(f'answer: accu: {accu_score}, f1: {f1_score}, prec: {precision_score}, recall: {recall_score}. resptime: mae: {mae}, mape: {mape}. fail rate: {fail_rate}')
    return [accu_score, f1_score, precision_score, recall_score, mae, mape, fail_rate]

config = {
    'select_type': None,
    'user_id': None,
    'group': None,
    'user_list': None,
    'encode': True,
    'stimuli_feature': 0, # 0-pretrained model extract feature: 25x2048, 1-handselected feature: 5x5
    'chars': "0123456789≡(mod) ", 
    'MaxMathLen': 11,
    'max_time_set': 10,
    'max_step': 60,
    'refresh_frequency': 5, # try different frequency
    'noise_delta': 0.001,
    'stimuli_discounter': 0.01, 
    'dataset_all_path': None,
    'train_dataset_path': None,
    'test_dataset_path': None,
    'test_result_name': 'test_result',
    'feedback_dataset_path': 'stimuli',
    'env_type': None,
    'train_step': 1000000, # 100000
    'test_epi_num': None,
    'test_format': None,
    'log_dir': None,
    'save_eval_file': True,
    'vis_eval': False,
}


def eval_user(config):
    config['select_type'] = 'user'
    # if config['encode'] == True:
    config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_ind_user_encode_norm.csv'
    # else:
    #     config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_ind_user_norm.csv'
    train_dataset_all = pd.read_csv(config['train_dataset_path'])
    user_all = _get_user_id(train_dataset_all)
    print('user_all: ', user_all)

    new_head = list(train_dataset_all.columns.values)[:-3]+['est_answer','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['user_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    model_metrics_all_svm = pd.DataFrame(columns = ['user_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    
    save_list([['user_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate']], 'rl_user_metrics.csv')
    save_list([['user_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate']], 'svm_user_metrics.csv')
    save_list([new_head], 'timecare~25k_rl_estimate_train_ind_user.csv')
    save_list([new_head], 'timecare~25k_rl_estimate_test_ind_user.csv')
    for counter_i, user_id in enumerate(user_all):
        config['log_dir'] = 'user_model_pure/' + str(int(user_id)) + '/' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '/'
        if os.path.exists('user_model_pure/') == False:
            os.mkdir('user_model_pure/')
        if os.path.exists('user_model_pure/' + str(int(user_id))) == False:
            os.mkdir('user_model_pure/' + str(int(user_id)))
        config['user_id'] = user_id
        config['env_type'] = 'train'
        config['dataset_all_path'] = config['train_dataset_path']
        train(config)

        config['env_type'] = 'test'
        config['test_format'] = 'all'

        # if config['encode'] == True:
        config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_train_ind_user_encode_norm.csv'
        # else:
            # config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_train_ind_user_norm.csv'
        config['test_result_name'] = 'train_pred_result'
        config['dataset_all_path'] = config['test_dataset_path']
        test_dataset_all = pd.read_csv(config['test_dataset_path'])
        config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], config['user_id'], None, config['max_time_set'])
        data_set_train = _get_user_data(test_dataset_all, config['select_type'], config['user_id'], None, config['max_time_set'])
        user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
        train_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
        train_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
        answer_train_estimated, resptime_train_estimated = agent_answer, agent_resptime

        # if config['encode'] == True:
        config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_ind_user_encode_norm.csv'
        # else:
            # config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_ind_user_norm.csv'
        config['test_result_name'] = 'test_pred_result'
        config['dataset_all_path'] = config['test_dataset_path']
        test_dataset_all = pd.read_csv(config['test_dataset_path'])
        config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], config['user_id'], None, config['max_time_set'])
        data_set_test = _get_user_data(test_dataset_all, config['select_type'], config['user_id'], None, config['max_time_set'])
        user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
        test_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
        test_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
        answer_test_estimated, resptime_test_estimated = agent_answer, agent_resptime

        # model_metrics_all.loc[counter_i] = [user_id] + train_metric + test_metric
        # model_metrics_all_svm.loc[counter_i] = [user_id] + train_metric_svm + test_metric_svm

        model_metrics_rl = [user_id] + train_metric + test_metric
        model_metrics_svm = [user_id] + train_metric_svm + test_metric_svm

        save_list([model_metrics_rl], 'rl_user_metrics.csv')
        save_list([model_metrics_svm], 'svm_user_metrics.csv')

        # dataset_train_piece = pd.DataFrame(np.concatenate((np.array(data_set_train)[:,:-3],answer_train_estimated,resptime_train_estimated),axis=1), columns=new_head)
        # dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test)[:,:-3],answer_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
        # estimated_dataset_train = pd.concat([estimated_dataset_train, dataset_train_piece],ignore_index=True)
        # estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)

        dataset_train_piece_arr = np.concatenate((np.array(data_set_train)[:,:-3],answer_train_estimated,resptime_train_estimated),axis=1)
        dataset_test_piece_arr = np.concatenate((np.array(data_set_test)[:,:-3],answer_test_estimated,resptime_test_estimated),axis=1)

        save_list(dataset_train_piece_arr, 'timecare~25k_rl_estimate_train_ind_user.csv')
        save_list(dataset_test_piece_arr, 'timecare~25k_rl_estimate_test_ind_user.csv')
    
    # estimated_dataset_train.to_csv('timecare~25k_rl_estimate_train_ind_user.csv',index=False)
    # estimated_dataset_test.to_csv('timecare~25k_rl_estimate_test_ind_user.csv',index=False)
    # model_metrics_all.to_csv('rl_user_metrics.csv',index=False)
    # model_metrics_all_svm.to_csv('svm_user_metrics.csv',index=False)

def eval_group(config):
    config['select_type'] = 'group'
    # if config['encode'] == True:
    config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_group_encode_norm.csv'
    # else:
        # config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_group_norm.csv'
    train_dataset_all = pd.read_csv(config['train_dataset_path'])
    group_all = _get_group_id(train_dataset_all)
    print('group_all: ', group_all)

    new_head = list(train_dataset_all.columns.values)[:-3]+['est_answer','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['group_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    model_metrics_all_svm = pd.DataFrame(columns = ['group_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    for counter_i, group_name in enumerate(group_all):
        config['log_dir'] = 'group_model_pure/' + group_name + '/' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '/'
        if os.path.exists('group_model_pure/') == False:
            os.mkdir('group_model_pure/')
        if os.path.exists('group_model_pure/' + group_name) == False:
            os.mkdir('group_model_pure/' + group_name)
        config['group'] = group_name
        config['env_type'] = 'train'
        config['dataset_all_path'] = config['train_dataset_path']
        train(config)

        config['env_type'] = 'test'
        config['test_format'] = 'all'
        # if config['encode'] == True:
        config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_train_group_encode_norm.csv'
        # else:
            # config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_train_group_norm.csv'
        config['test_result_name'] = 'train_pred_result'
        config['dataset_all_path'] = config['test_dataset_path']
        test_dataset_all = pd.read_csv(config['test_dataset_path'])
        config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], None, config['group'], config['max_time_set'])
        data_set_train = _get_user_data(test_dataset_all, config['select_type'], None, config['group'], config['max_time_set'])
        user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
        train_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
        train_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
        answer_train_estimated, resptime_train_estimated = agent_answer, agent_resptime

        # if config['encode'] == True:
        config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_group_encode_norm.csv'
        # else:
        #     config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_group_norm.csv'
        config['test_result_name'] = 'test_pred_result'
        config['dataset_all_path'] = config['test_dataset_path']
        test_dataset_all = pd.read_csv(config['test_dataset_path'])
        config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], None, config['group'], config['max_time_set'])
        data_set_test = _get_user_data(test_dataset_all, config['select_type'], None, config['group'], config['max_time_set'])
        user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
        test_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
        test_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
        answer_test_estimated, resptime_test_estimated = agent_answer, agent_resptime

        model_metrics_all.loc[counter_i] = [group_name] + train_metric + test_metric
        model_metrics_all_svm.loc[counter_i] = [group_name] + train_metric_svm + test_metric_svm

        dataset_train_piece = pd.DataFrame(np.concatenate((np.array(data_set_train)[:,:-3],answer_train_estimated,resptime_train_estimated),axis=1), columns=new_head)
        dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test)[:,:-3],answer_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
        estimated_dataset_train = pd.concat([estimated_dataset_train, dataset_train_piece],ignore_index=True)
        estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)
    
    estimated_dataset_train.to_csv('timecare~25k_rl_estimate_train_group.csv',index=False)
    estimated_dataset_test.to_csv('timecare~25k_rl_estimate_test_group.csv',index=False)
    model_metrics_all.to_csv('rl_group_metrics.csv',index=False)
    model_metrics_all_svm.to_csv('svm_group_metrics.csv',index=False)

def eval_general_all(config):
    config['select_type'] = 'all'
    # if config['encode'] == True:
    config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_general_encode_norm.csv'
    # else:
        # config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_general_norm.csv'
    train_dataset_all = pd.read_csv(config['train_dataset_path'])

    new_head = list(train_dataset_all.columns.values)[:-3]+['est_answer','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['general','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    model_metrics_all_svm = pd.DataFrame(columns = ['general','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    
    config['log_dir'] = 'general_model_pure/' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '/'
    if os.path.exists('general_model_pure/') == False:
        os.mkdir('general_model_pure/')
    config['env_type'] = 'train'
    config['dataset_all_path'] = config['train_dataset_path']
    train(config)

    config['env_type'] = 'test'
    config['test_format'] = 'all'
    # if config['encode'] == True:
    config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_train_general_encode_norm.csv'
    # else:
        # config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_train_general_norm.csv'
    config['test_result_name'] = 'train_pred_result'
    config['dataset_all_path'] = config['test_dataset_path']
    test_dataset_all = pd.read_csv(config['test_dataset_path'])
    config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], None, None, config['max_time_set'])
    data_set_train = _get_user_data(test_dataset_all, config['select_type'], None, None, config['max_time_set'])
    user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
    train_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
    train_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
    answer_train_estimated, resptime_train_estimated = agent_answer, agent_resptime
    # if config['encode'] == True:
    config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_general_encode_norm.csv'
    # else:
    #     config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_general_norm.csv'
    config['test_result_name'] = 'test_pred_result'
    config['dataset_all_path'] = config['test_dataset_path']
    test_dataset_all = pd.read_csv(config['test_dataset_path'])
    config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], None, None, config['max_time_set'])
    data_set_test = _get_user_data(test_dataset_all, config['select_type'], None, None, config['max_time_set'])
    user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
    test_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
    test_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
    answer_test_estimated, resptime_test_estimated = agent_answer, agent_resptime

    model_metrics_all.loc[0] = ['general'] + train_metric + test_metric
    model_metrics_all_svm.loc[0] = ['general'] + train_metric_svm + test_metric_svm

    dataset_train_piece = pd.DataFrame(np.concatenate((np.array(data_set_train)[:,:-3],answer_train_estimated,resptime_train_estimated),axis=1), columns=new_head)
    dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test)[:,:-3],answer_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
    estimated_dataset_train = pd.concat([estimated_dataset_train, dataset_train_piece],ignore_index=True)
    estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)
    
    if config['encode'] == True:
        estimated_dataset_train.to_csv('timecare~25k_rl_estimate_train_general_encode.csv',index=False)
        estimated_dataset_test.to_csv('timecare~25k_rl_estimate_test_general_encode.csv',index=False)
        model_metrics_all.to_csv('rl_general_metrics_encode.csv',index=False)
        model_metrics_all_svm.to_csv('svm_general_metrics_encode.csv',index=False)
    else:
        estimated_dataset_train.to_csv('timecare~25k_rl_estimate_train_general_noencode.csv',index=False)
        estimated_dataset_test.to_csv('timecare~25k_rl_estimate_test_general_noencode.csv',index=False)
        model_metrics_all.to_csv('rl_general_metrics_noencode.csv',index=False)
        model_metrics_all_svm.to_csv('svm_general_metrics_noencode.csv',index=False)

def ablation_study_frequency(config):
    config['select_type'] = 'all'
    # if config['encode'] == True:
    config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_general_encode_norm.csv'
    # else:
    #     config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_general_norm.csv'
    train_dataset_all = pd.read_csv(config['train_dataset_path'])

    new_head = list(train_dataset_all.columns.values)[:-3]+['est_answer','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['refresh_fre','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    model_metrics_all_svm = pd.DataFrame(columns = ['refresh_fre','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    for counter_i, refresh_fre in enumerate([1,3,5,10,30]):
        config['log_dir'] = 'ablation_study_frequency/' + str(int(refresh_fre)) + '/' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '/'
        if os.path.exists('ablation_study_frequency/' + str(int(refresh_fre))) == False:
            os.mkdir('ablation_study_frequency/' + str(int(refresh_fre)))
        config['refresh_frequency'] = refresh_fre
        config['env_type'] = 'train'
        config['dataset_all_path'] = config['train_dataset_path']
        train(config)

        config['env_type'] = 'test'
        config['test_format'] = 'all'
    
        # if config['encode'] == True:
        config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_general_encode_norm.csv'
        # else:
        #     config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_general_norm.csv'
        config['test_result_name'] = 'test_pred_result'
        config['dataset_all_path'] = config['test_dataset_path']
        test_dataset_all = pd.read_csv(config['test_dataset_path'])
        config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], None, None, config['max_time_set'])
        data_set_test = _get_user_data(test_dataset_all, config['select_type'], None, None, config['max_time_set'])
        user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
        test_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
        test_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
        answer_test_estimated, resptime_test_estimated = agent_answer, agent_resptime

        model_metrics_all.loc[counter_i] = [refresh_fre] + test_metric
        model_metrics_all_svm.loc[counter_i] = [refresh_fre] + test_metric_svm

        dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test)[:,:-3],answer_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
        estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)
    
    estimated_dataset_test.to_csv('timecare~25k_rl_estimate_test_study_frequency.csv',index=False)
    model_metrics_all.to_csv('rl_study_frequency_metrics.csv',index=False)
    model_metrics_all_svm.to_csv('svm_study_frequency_metrics.csv',index=False)

def ablation_study_discounter(config):
    config['select_type'] = 'all'
    # if config['encode'] == True:
    config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_general_encode_norm.csv'
    # else:
    #     config['train_dataset_path'] = '../svm_model/timecare~25k_estimate_train_general_norm.csv'
    train_dataset_all = pd.read_csv(config['train_dataset_path'])

    new_head = list(train_dataset_all.columns.values)[:-3]+['est_answer','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    model_metrics_all = pd.DataFrame(columns = ['discounter','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    model_metrics_all_svm = pd.DataFrame(columns = ['discounter','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate'])
    for counter_i, discounter in enumerate([0.0001,0.001,0.01,0.1]):
    # for counter_i, discounter in enumerate([0.0001]):
        config['log_dir'] = 'ablation_study_discounter/' + str(discounter) + '/' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '/'
        # config['log_dir'] = 'ablation_study_discounter/' + str(discounter) + '/' + '22_06_28_14_30_44' + '/'
        if os.path.exists('ablation_study_discounter/' + str(discounter)) == False:
            os.mkdir('ablation_study_discounter/' + str(discounter))
        config['stimuli_discounter'] = discounter
        config['env_type'] = 'train'
        config['dataset_all_path'] = config['train_dataset_path']
        train(config)

        config['env_type'] = 'test'
        config['test_format'] = 'all'
        # if config['encode'] == True:
        config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_general_encode_norm.csv'
        # else:
        #     config['test_dataset_path'] = '../svm_model/timecare~25k_estimate_test_general_norm.csv'
        config['test_result_name'] = 'test_pred_result'
        config['dataset_all_path'] = config['test_dataset_path']
        test_dataset_all = pd.read_csv(config['test_dataset_path'])
        # test_dataset_all = pd.DataFrame(np.array(test_dataset_all)[:10],columns=test_dataset_all.columns.values)
        config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], None, None, config['max_time_set'])
        data_set_test = _get_user_data(test_dataset_all, config['select_type'], None, None, config['max_time_set'])
        user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
        test_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
        test_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
        answer_test_estimated, resptime_test_estimated = agent_answer, agent_resptime

        model_metrics_all.loc[counter_i] = [discounter] + test_metric
        model_metrics_all_svm.loc[counter_i] = [discounter] + test_metric_svm

        dataset_test_piece = pd.DataFrame(np.concatenate((np.array(data_set_test)[:,:-3],answer_test_estimated,resptime_test_estimated),axis=1), columns=new_head)
        estimated_dataset_test = pd.concat([estimated_dataset_test, dataset_test_piece],ignore_index=True)
    
    estimated_dataset_test.to_csv('timecare~25k_rl_estimate_test_study_discounter.csv',index=False)
    model_metrics_all.to_csv('rl_study_discounter_metrics.csv',index=False)
    model_metrics_all_svm.to_csv('svm_study_discounter_metrics.csv',index=False)

def eval_leave_one_participant_out_group(config):
    
    raw_dataset_path = '../svm_model/timecare~25k_raw_filtered.csv'
    dataset_all = pd.read_csv(raw_dataset_path)
    group_all = _get_group_id(dataset_all)
    group_all.sort()
    user_all = _get_user_id(dataset_all)
    user_all.sort()
    group_user_dict = _get_group_users(raw_dataset_path)
    print('user_all: ', user_all, 'group_all: ', group_all)

    new_head = list(dataset_all.columns.values)+['est_answer','est_resptime']
    estimated_dataset_train = pd.DataFrame(columns = new_head)
    estimated_dataset_test = pd.DataFrame(columns = new_head)
    
    save_list([['user_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate']], 'rl_lopo_metrics.csv')
    save_list([['user_id','train_accu','train_f1','train_precision','train_recall','train_mae', 'train_mape','train_fail_rate','test_accu','test_f1','test_precision','test_recall','test_mae','test_mape','test_fail_rate']], 'svm_lopo_metrics.csv')
    save_list([new_head], 'timecare~25k_rl_estimate_train_lopo.csv')
    save_list([new_head], 'timecare~25k_rl_estimate_test_lopo.csv')

    if os.path.exists('lopo_model_pure/') == False:
        os.mkdir('lopo_model_pure/')

    for counter_i,group_name in enumerate(group_all):
        user_all_per_group = group_user_dict[group_name]
        if os.path.exists('lopo_model_pure/' + group_name) == False:
            os.mkdir('lopo_model_pure/' + group_name)
        for counter_j,user_id in enumerate(user_all_per_group):
            config['log_dir'] = 'lopo_model_pure/' + group_name + '/' + str(int(user_id)) + '/' + datetime.now().strftime('%y_%m_%d_%H_%M_%S') + '/'
            if os.path.exists('lopo_model_pure/' + group_name + '/' + str(int(user_id))) == False:
                os.mkdir('lopo_model_pure/' + group_name + '/' + str(int(user_id)))
            # user_group_train = user_all_per_group.copy()
            # del user_group_train[user_group_train.index(user_id)]
            config['select_type'] = 'all'
            # config['user_list'] = user_group_train
            config['env_type'] = 'train'
            config['train_dataset_path'] = '../svm_model/lopo_model/'+group_name+'/'+str(int(user_id))+'/trainset.csv'
            config['dataset_all_path'] = config['train_dataset_path']
            train(config)

            config['select_type'] = 'all'
            config['env_type'] = 'test'
            config['test_format'] = 'all'

            config['test_dataset_path'] = '../svm_model/lopo_model/'+group_name+'/'+str(int(user_id))+'/trainset.csv'
            config['test_result_name'] = 'train_pred_result'
            config['dataset_all_path'] = config['test_dataset_path']
            test_dataset_all = pd.read_csv(config['test_dataset_path'])
            config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], config['user_id'], None, config['max_time_set'])
            data_set_train = _get_user_data(test_dataset_all, config['select_type'], config['user_id'], None, config['max_time_set'])
            user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
            train_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
            train_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
            answer_train_estimated, resptime_train_estimated = agent_answer, agent_resptime

            config['test_dataset_path'] = '../svm_model/lopo_model/'+group_name+'/'+str(int(user_id))+'/testset.csv'
            config['test_result_name'] = 'test_pred_result'
            config['dataset_all_path'] = config['test_dataset_path']
            test_dataset_all = pd.read_csv(config['test_dataset_path'])
            config['test_epi_num'] = _get_epi_num(test_dataset_all, config['select_type'], config['user_id'], None, config['max_time_set'])
            data_set_test = _get_user_data(test_dataset_all, config['select_type'], config['user_id'], None, config['max_time_set'])
            user_answer, user_resptime, est_answer, est_resptime, agent_answer, agent_resptime, fail_rate = test(config)
            test_metric = eval_metrics(user_answer, agent_answer, user_resptime, agent_resptime)
            test_metric_svm = eval_metrics(user_answer, est_answer, user_resptime, est_resptime)
            answer_test_estimated, resptime_test_estimated = agent_answer, agent_resptime

            model_metrics_rl = [user_id] + train_metric + test_metric
            model_metrics_svm = [user_id] + train_metric_svm + test_metric_svm

            save_list([model_metrics_rl], 'rl_lopo_metrics.csv')
            save_list([model_metrics_svm], 'svm_lopo_metrics.csv')

            dataset_train_piece_arr = np.concatenate((np.array(data_set_train)[:,:-3],answer_train_estimated,resptime_train_estimated),axis=1)
            dataset_test_piece_arr = np.concatenate((np.array(data_set_test)[:,:-3],answer_test_estimated,resptime_test_estimated),axis=1)

            dataset_train_piece = pd.DataFrame(dataset_train_piece_arr, columns=new_head)
            dataset_test_piece = pd.DataFrame(dataset_test_piece_arr, columns=new_head)

            dataset_train_piece.to_csv('lopo_model_pure/'+group_name+'/'+str(int(user_id))+'/trainset.csv',index=False)
            dataset_test_piece.to_csv('lopo_model_pure/'+group_name+'/'+str(int(user_id))+'/testset.csv',index=False)

            save_list(dataset_train_piece_arr, 'timecare~25k_rl_estimate_train_lopo.csv')
            save_list(dataset_test_piece_arr, 'timecare~25k_rl_estimate_test_lopo.csv')  




if args.type == 'eu':
    eval_user(config)
elif args.type == 'eg':
    eval_group(config)
elif args.type == 'ea':
    eval_general_all(config)
elif args.type == 'el':
    eval_leave_one_participant_out_group(config)
elif args.type == 'abf':
    ablation_study_frequency(config)
elif args.type == 'abd':
    ablation_study_discounter(config)
elif args.type == 'abdf':
    ablation_study_discounter(config)
    ablation_study_frequency(config)
elif args.type == 'test':
    config['encode'] = True
    eval_general_all(config)
    config['encode'] = False
    eval_general_all(config)

