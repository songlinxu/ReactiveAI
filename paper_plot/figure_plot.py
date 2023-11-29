
import seaborn as sns
# sns.set_theme(style="darkgrid")
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
custom_params = {}
# custom_params = {"axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale = 1.2)

import numpy as np
from numpy.random import seed
seed(4)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import matplotlib.ticker as ticker
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import norm
from math import sqrt

import os, sys, time

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy.stats import pearsonr
from scipy.interpolate import make_interp_spline, BSpline
from stable_baselines3.common.results_plotter import load_results, ts2xy



# Tools ----------------------------------------------------------------------------------------------------------------------------------
def _get_user_id(dataset_table):
    return list(set(dataset_table['user_id']))


def _get_group_id(dataset_table):
    return list(set(dataset_table['group']))

def _get_user2group_dict(dataset_table):
    group_all = _get_group_id(dataset_table)
    user_all = _get_user_id(dataset_table)
    group_dict = {}
    for i,user_id in enumerate(user_all):
        data_user = dataset_table[dataset_table['user_id']==user_id]
        group_dict[user_id] = data_user['group'].values[0]
    return group_dict

def read_text(text_path, padding = False):
    with open(text_path, "r") as fr:
        lines = fr.readlines()
        new_lines = []
        for line in lines:
            new_lines.append(list(np.float_(line.strip().split(','))))
        if padding == False:
            return new_lines
        else:
            return padding_list(new_lines, get_max_len(new_lines))

def generate_stimuli_trajectory(action_list, delta_proba):
    stimuli_trajectory = []
    current_accumulation = 0
    for i,action in enumerate(action_list):
        current_accumulation += action * delta_proba
        stimuli_trajectory.append(current_accumulation)
    return stimuli_trajectory

def integrate_trajectory(base_trace, stimuli_trace, stimuli_discounter = 0.01):
    # print(len(base_trace), len(stimuli_trace))
    # assert len(base_trace) == len(stimuli_trace)
    final_trace = np.array(base_trace[:len(stimuli_trace)]) + np.array(stimuli_trace) * stimuli_discounter
    return final_trace


def generate_base_trajectory(resptime, answer_proba, frequency = 10, init_interval = 20, delta = 0.01, max_time_set = 10):
    # print('resptime: ', resptime)
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
    if len(final_signal) < int(max_time_set*frequency):
        trajectory = np.zeros((int(max_time_set*frequency))) + final_signal[-1]
        trajectory[:len(final_signal)] = final_signal
    else:
        trajectory = final_signal
    # print('trajectory: ', trajectory)

    return trajectory, answer, delta_proba


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
    # print(y_init_end)
    y = y * (abs(y_end-y_start)/abs(y_init_end-y_init_start)) + y_start - y_init_start
    return x, y

def wiener_curve(x_start, x_end, y_start, y_end, frequency = 10, delta = 0.01):
    step_num = int(abs(x_end-x_start)*frequency)
    dt = abs(y_end-y_start)/step_num
    noise_array = brownian(0, step_num, dt, delta)
    return noise_array

def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
    np.cumsum(r, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def _get_group_users(dataset_path):
    raw_dataset = pd.read_csv(dataset_path)
    group_all = list(set(raw_dataset['group']))
    group_user_dict = {}
    for i,group_name in enumerate(group_all):
        data_each_group = raw_dataset[raw_dataset['group']==group_name]
        group_user_dict[group_name] = list(set(data_each_group['user_id']))
    return group_user_dict

def _get_user_group_dict(dataset_path):
    raw_dataset = pd.read_csv(dataset_path)
    user_all = list(set(raw_dataset['user_id']))
    user_group_dict = {}
    for i,user_name in enumerate(user_all):
        data_each_user = raw_dataset[raw_dataset['user_id']==user_name]
        user_group_dict[user_name] = data_each_user['group'].values[0]
    print(user_group_dict)
    return user_group_dict

def turn_table_resptime(datatable, filter_type = 'answer', target_type = 'resptime'):
    for i in range(len(datatable)):
        if datatable.at[i,filter_type] == 0:
            datatable.at[i,target_type] = float(-datatable.at[i,target_type])
    return datatable.copy()

def eval_metrics(y_answer_test, y_answer_predicted, y_resptime_test, y_resptime_predicted, return_type = 'print'):
    y_answer_test = np.float_(y_answer_test).flatten()
    y_answer_predicted = np.float_(y_answer_predicted).flatten()
    y_resptime_test = np.float_(y_resptime_test).flatten()
    y_resptime_predicted = np.float_(y_resptime_predicted).flatten()
    len_old = len(y_answer_test)

    indices = np.where(y_answer_predicted != -1)[0]
    
    y_answer_test = y_answer_test[indices]
    y_answer_predicted = y_answer_predicted[indices]
    y_resptime_test = y_resptime_test[indices]
    y_resptime_predicted = y_resptime_predicted[indices]
    len_new = len(indices)
    
    failrate = 1 - len_new/len_old

    
    mape = metrics.mean_absolute_percentage_error(y_resptime_test, y_resptime_predicted)
    pearson_corre, _ = pearsonr(y_resptime_test, y_resptime_predicted)
    
    mape_str = str(np.around(mape, decimals=4))
    if return_type == 'print':
        # return accu_score_str + ' & ' + precision_score_str + ' & ' + recall_score_str + ' & ' + f1_score_str + ' & ' + mape_str + ' & 0.\\'
        return mape_str + ' & 0.\\'
    elif return_type == 'dict':
        # return {'accuracy': accu_score, 'precision': precision_score, 'recall': recall_score, 'f1': f1_score, 'mape': mape, 'failrate': failrate}
        return {'mape': mape, 'failrate': failrate, 'pearson': pearson_corre}

def table_plot(model_type, select_type, target_select, return_type, tail = ''):
    assert select_type == 'general' or select_type == 'group' or select_type == 'ind_user' or select_type == 'lopo'
    if model_type == 'svm':
        dataset_path = '../svm_model/timecare~25k_estimate_test_' + select_type + tail + '_encode_norm.csv'
    elif model_type == 'drl_ddm':
        # dataset_path = '../rl_model/result/DRL_DDM_Fine_0.01/timecare~25k_rl_estimate_test_' + select_type + tail + '.csv'
        dataset_path = '../rl_model/result/DRL_DDM_Fine_Tune/timecare~25k_rl_estimate_test_' + select_type + tail + '.csv'
        # dataset_path = '../rl_model/result/DRL_DDM_Fine_0.01/timecare~25k_rl_estimate_test_lopo' + '.csv'
    elif model_type == 'drl_only':
        dataset_path = '../rl_model/result/DRL_Only/timecare~25k_rl_estimate_test_' + select_type + tail + '.csv'
    dataset = pd.read_csv(dataset_path)
    dataset = pd.DataFrame(np.array(dataset),columns = dataset.columns.values)
    # print('new dataset len: ', len(dataset))
    result = dict()
    if target_select == 'general':
        result['general'] = eval_metrics(dataset['answer'], dataset['est_answer'], dataset['resptime'], dataset['est_resptime'], return_type)
        # print(f'{model_type} model, {select_type} dataset, {target_select} results-accu, prec, recall, f1, mape: {result}')
    elif target_select == 'group':
        group_all = list(set(dataset['group']))
        for group in group_all:
            dataset_each_group = dataset[dataset['group']==group]
            dataset_each_group = pd.DataFrame(np.array(dataset_each_group), columns = dataset.columns.values)
            result[group] = eval_metrics(dataset_each_group['answer'], dataset_each_group['est_answer'], dataset_each_group['resptime'], dataset_each_group['est_resptime'], return_type) 
            # print('group: ', group)
            # print(f'{model_type} model, {select_type} dataset, {target_select} results-accu, prec, recall, f1, mape: {group}, {result[group]}')
    elif target_select == 'ind_user':
        user_all = list(set(dataset['user_id']))
        for user_id in user_all:
            dataset_each_user = dataset[dataset['user_id']==user_id]
            dataset_each_user = pd.DataFrame(np.array(dataset_each_user), columns = dataset.columns.values)
            result[user_id] = eval_metrics(dataset_each_user['answer'], dataset_each_user['est_answer'], dataset_each_user['resptime'], dataset_each_user['est_resptime'], return_type) 
            # print(f'{model_type} model, {select_type} dataset, {target_select}, user id: {user_id}, results-accu, prec, recall, f1, mape: {user_id}, {result[user_id]}')
    return result

# Dataset Exploration ----------------------------------------------------------------------------------------------------------------------------------

def plot_dataset_four_type(dataset_path):
    dataset = pd.read_csv(dataset_path)
    dataset = dataset[(dataset['session']=='formal')&(dataset['block_id']!=1)]
    fig, ax = plt.subplots(constrained_layout=True)
    ax2 = ax.axes.twinx()
    sns.lineplot(data = dataset, x = 'block_id', y = 'rela_1_resptime', ax = ax, label = 'response time', color = 'b', alpha = 1)
    sns.lineplot(data = dataset, x = 'block_id', y = 'rela_1_accuracy', ax = ax, label = 'accuracy', color = 'orange', alpha = 1)
    sns.lineplot(data = dataset, x = 'block_id', y = 'rela_focus', ax = ax2, label = 'attention', color = 'g', alpha = 1)
    sns.lineplot(data = dataset, x = 'block_id', y = 'rela_anxiety', ax = ax2, label = 'anxiety', color = 'r', alpha = 1)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    ax.set_xlabel('Block ID', fontsize=18)
    ax.set_ylabel('')
    ax2.set_ylabel('')
   
    # plt.legend(loc='upper center',ncol=1)
    xticks = ['2','3','4','5']
    ax.set_xticks([r+2 for r in range(len(xticks))], xticks)
    plt.show()



def plot_dataset_distribution(dataset_path, target_type, compare_type, dist_type = 'line'):
    assert compare_type in ['group','block_id','day']
    assert dist_type in ['line','step']
    dataset = pd.read_csv(dataset_path)
    dataset = dataset[(dataset['session']=='formal')&(dataset['block_id']!=1)]
    fig, ax = plt.subplots(constrained_layout=True) 
    if dist_type == 'line':
        sns.displot(dataset, x=target_type, hue=compare_type, kind="kde", fill=True, ax = ax)
    else:
        ax=sns.histplot(dataset, x=target_type, hue=compare_type, fill=True, element="step")
    # ax.set_xlabel('Relative Anxiety Change') 
    # plt.legend(loc='upper right')
    # ax.spines['left'].set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.show()

def plot_dataset_bar_seaborn(dataset_path, target_type, x_type, z_type):
    # fig, ax = plt.subplots(constrained_layout=True) 
    fig, ax = plt.subplots(figsize=(6,3), constrained_layout=True) 
    dataset = pd.read_csv(dataset_path)
    dataset = dataset[(dataset['session']=='formal')&(dataset['block_id']!=1)]
    group_list = ['none','static','random','rule']
    block_list = [2,3,4,5]
    date_list = ['d1','d2']
    PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
    }
    # sns.barplot(x="group", y=target_type, hue="block_id", data=dataset, ci=None)
    ax = sns.boxplot(x=x_type, y=target_type, hue = z_type, data=dataset, flierprops = dict(markerfacecolor='w', marker='o'), width=0.75, palette="Blues")
    # ax = sns.boxplot(x=x_type, y=target_type, hue = z_type, data=dataset, flierprops = dict(markerfacecolor='w', marker='o'), width=0.5, color='white', palette="Blues", **PROPS)
    # ax = sns.boxplot(x=x_type, y=target_type, data=dataset, flierprops = dict(markerfacecolor='w', marker='o'), width=0.5, color='white', **PROPS)
    # ax = sns.stripplot(x=x_type, y=target_type, hue = z_type, data=dataset, color="orange", jitter=0.2, size=2.5, alpha=1)
    # ax = sns.stripplot(x=x_type, y=target_type, data=dataset, color="orange", jitter=0.2, size=2.5, alpha=1)
    # sns.boxplot(x="group", y=target_type, hue="block_id", data=dataset)
    # sns.boxplot(x="block_id", y=target_type, hue="group", data=dataset)
    # sns.catplot(x="block_id", y=target_type, hue="group", col='day', data=dataset, kind="box")
    plt.legend(loc='upper center',ncol=4,bbox_to_anchor=(0.5,1.3))
    # plt.legend([])
    sns.despine(offset = 5, trim = True)
    plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    plt.setp(ax.lines, color='k')
    plt.show()



# Math Answer Agent ----------------------------------------------------------------------------------------------------------------------------------

def plot_math_agent(math_folder_path, interval = 20):
    neuron_list = [32,64,128,256]
    circle_color = ['g','gray','r','b']
    alpha_list = [0.5,0.5,0.5,0.5]
    # neuron_list = [256]
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for i,neuron in enumerate(neuron_list):
        history_path = math_folder_path + '/model_history_' + str(neuron) + '.csv'
        history_data = pd.read_csv(history_path)
        history_data['index'] = history_data.index
        # history_data = history_data[::interval]
        plt.plot(history_data['loss'], c=circle_color[i], alpha=1, zorder=2)
    for i,neuron in enumerate(neuron_list):
        history_path = math_folder_path + '/model_history_' + str(neuron) + '.csv'
        history_data = pd.read_csv(history_path)
        history_data['index'] = history_data.index
        history_data = history_data[::interval]
        # plt.plot(history_data['loss'], c=circle_color[i])
        plt.scatter(x=history_data.index, y=history_data['loss'], s=history_data['accuracy']*5000, alpha=alpha_list[i], c=circle_color[i],edgecolors='none',label=neuron,zorder=1) # 5000
        # plt.scatter(x=history_data.index, y=history_data['loss'], marker='x', s=20, c='black')
    for i,neuron in enumerate(neuron_list):
        history_path = math_folder_path + '/model_history_' + str(neuron) + '.csv'
        history_data = pd.read_csv(history_path)
        history_data = history_data[::interval]
        plt.scatter(x=history_data.index, y=history_data['loss'], marker='x', s=20, c='black', zorder=3)
    accu_list = [0.25,0.5,0.75,1]
    x_list = [40,53,70,90]
    for i,accuracy in enumerate(accu_list):
        # interval = 5
        # radius = accuracy * 40 / 2
        # radius_last = accu_list[i-1] * 40 / 2
        # x = 40+radius+interval+radius_last if i !=0 else 40
        plt.scatter(x=x_list[i], y=1.95, s=accuracy*5000, alpha=alpha_list[0], c='gray',edgecolors='none',zorder=4) # 5000
    plt.text(50,1.5,'% of Prediction Accuracy',zorder = 5)
    plt.text(0,0.28,'Neuron',zorder = 5)
    sns.despine(offset=10, trim=True)
    plt.ylim([-0.4, 2.5])
    legend = plt.legend(loc='lower left', ncol=1, fontsize=10) # bbox_to_anchor=(0.5, 1.1), 
    legend.legendHandles[0]._sizes = [50]
    legend.legendHandles[1]._sizes = [50]
    legend.legendHandles[2]._sizes = [50]
    legend.legendHandles[3]._sizes = [50]
    # ax.spines.top.set_visible(True)
    # ax.spines.top.set_bounds(40, 80)
    # ax.spines.top.set_position(('data', 1.75))
    # ax.tick_params(top=True)
    # twin = ax.twiny()
    # twin.tick_params(axis='x')
    # twin.set_xlim(40, 80)
    axins1 = inset_axes(ax, width="45%", height="10%", bbox_to_anchor=(-0.12,0.08,1,1), bbox_transform=ax.transAxes) # #, bbox_to_anchor=(1.05,0,1,1)
    # axins1 = inset_axes(ax, loc='upper right', bbox_to_anchor=(0.5,1.1)) # #, bbox_to_anchor=(1.05,0,1,1)
    # axins1.set_xlim(0.25,1)
    # axins1.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    axins1.set_xticks([0,0.25,0.6,1], ['25','50','75','100'])
    # axins1.set_xlabel('accuracy')
    axins1.xaxis.set_ticks_position("bottom")
    axins1.tick_params(direction='in')
    axins1.spines['left'].set_visible(False)
    axins1.axes.get_yaxis().set_visible(False)


    plt.show()

def plot_confusion_matrix(output_file):
    output_table = pd.read_csv(output_file)
    y_true, y_pred = np.array(output_table['answer']), np.array(output_table['pred'])
    # print(f'accuracy: {np.sum(y_true==y_pred)/len(y_pred)}')
    matrix = confusion_matrix(y_true, y_pred)
    # ax = sns.heatmap(matrix, annot=True, fmt="d") # , cmap="YlGnBu"
    ax = sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5, linecolor='w') # , cmap="YlGnBu"
    
    plt.show()


# SVM ----------------------------------------------------------------------------------------------------------------------------------

def plot_svm_math_feature(encode_result_path, string_result_path, num_result_path):
    result_1 = np.array(pd.read_csv(encode_result_path))[0][-6:-2]
    result_2 = np.array(pd.read_csv(string_result_path))[0][-6:-2]
    result_3 = np.array(pd.read_csv(num_result_path))[0][-6:-2]
    print('result_1: ', result_1, ' result_2: ', result_2, ' result_3: ', result_3)
    color_type_list = ['black','black']
    color_feature_list = ['cyan', 'springgreen', 'lightcoral']
    # color_feature_list = ['g', 'b', 'r']
    # color_feature_list = ['fuchsia', 'white', 'black']
    xticks = ['Accuracy','F1-Score','Precision','Recall']
    xticks_more = ['Accuracy','F1-Score','Precision','Recall','MAPE']
    fig, ax1 = plt.subplots(constrained_layout=True) 
    alpha = 0.8
    barWidth = 0.25
    br1 = np.arange(len(xticks))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    ax1.bar(br1, result_1, color = color_feature_list[0], alpha=alpha, edgecolor=color_type_list[0], width = barWidth, linewidth=1, label = 'feature')
    ax1.bar(br2, result_2, color = color_feature_list[1], alpha=alpha, edgecolor=color_type_list[0], width = barWidth, label = 'encode')
    ax1.bar(br3, result_3, color = color_feature_list[2], alpha=alpha, edgecolor=color_type_list[0], width = barWidth, label = 'digits')
    # ax1.bar(br1, result_1, color = color_feature_list[0], alpha=alpha, width = barWidth, linewidth=1, label = 'feature')
    # ax1.bar(br2, result_2, color = color_feature_list[1], alpha=alpha, width = barWidth, label = 'encode')
    # ax1.bar(br3, result_3, color = color_feature_list[2], alpha=alpha, width = barWidth, label = 'digits')
    ax1.set_xticks([r + barWidth for r in range(len(xticks_more))], xticks_more)
    # ax1.set_xlabel('Metrics') 
    ax1.set_ylabel('Choice Classification', color = color_type_list[0]) 
    ax1.tick_params(axis ='y', labelcolor = color_type_list[0])
    ax1.set_ylim([0,1])

    result_1 = np.array(pd.read_csv(encode_result_path))[0][-1:]
    result_2 = np.array(pd.read_csv(string_result_path))[0][-1:]
    result_3 = np.array(pd.read_csv(num_result_path))[0][-1:]
    print('result_1: ', result_1, ' result_2: ', result_2, ' result_3: ', result_3)
    xticks2 = ['mape']
    ax2 = ax1.axes.twinx()
    barWidth = 0.25
    br1 = [len(xticks)]
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    ax2.bar(br1, result_1, color = color_feature_list[0], alpha=alpha, edgecolor=color_type_list[1], width = barWidth, label = 'feature')
    ax2.bar(br2, result_2, color = color_feature_list[1], alpha=alpha, edgecolor=color_type_list[1], width = barWidth, label = 'encode')
    ax2.bar(br3, result_3, color = color_feature_list[2], alpha=alpha, edgecolor=color_type_list[1], width = barWidth, label = 'digits')
    # ax2.bar(br1, result_1, color = color_feature_list[0], alpha=alpha, width = barWidth, label = 'feature')
    # ax2.bar(br2, result_2, color = color_feature_list[1], alpha=alpha, width = barWidth, label = 'encode')
    # ax2.bar(br3, result_3, color = color_feature_list[2], alpha=alpha, width = barWidth, label = 'digits')
    # ax2.set_xticks([r + barWidth for r in range(len(xticks2))], xticks2)
    # ax2.set_xlabel('metrics') 
    ax2.set_ylabel('Response Time Estimation', color = color_type_list[1]) 
    ax2.tick_params(axis ='y', labelcolor = color_type_list[1])
    ax2.set_ylim([0.35,0.39])

    plt.axvline(x=4-barWidth, color='red', linestyle='--', alpha = 0.5)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.show()


# DRL ----------------------------------------------------------------------------------------------------------------------------------

def plot_all_train_return(root_path, model_type = 'pure', len_max = None, norm = None):
    assert model_type in ['pure','hybrid']
    fig, axs = plt.subplots(1,1,constrained_layout = True)
    all_user_models = os.listdir(root_path)
    if '.DS_Store' in all_user_models:
        del all_user_models[all_user_models.index('.DS_Store')]
    all_user_models.sort()
    for i,user_folder in enumerate(all_user_models):
        all_models = os.listdir(root_path+'/'+user_folder)
        if '.DS_Store' in all_models:
            del all_models[all_models.index('.DS_Store')]
        all_models.sort()
        # curve_data = pd.read_csv(root_path+'/'+user_folder+'/'+all_models[-1]+'/train_curve.csv')
        x, y = ts2xy(load_results(root_path+'/'+user_folder+'/'+all_models[-1]), 'timesteps')
        y = moving_average(y, window=50)
        x = x[len(x) - len(y):]
        if len_max != None:
            min_index = min(i for i,xi in enumerate(x) if xi > len_max)
        else:
            min_index = len(x)
        y = y/norm if norm != None else y
        plt.plot(x[:min_index],y[:min_index],alpha=1)
        print(len(x))
        print(x)
        # sns.lineplot(x='x',y='y',data=curve_data)
    # axs.set_ylim(-0.3,0.5)
    axs.set_xlabel('Step')
    axs.set_ylabel('Episode Reward')
    if model_type == 'pure':
        xticks = ['0','200000','400000','600000','800000','1000000']
        axs.set_xticks([(r)*200000 for r in range(len(xticks))], xticks)
    # else:
    #     xticks = ['0','20000','40000','60000','80000','100000']
    #     axs.set_xticks([(r)*20000 for r in range(len(xticks))], xticks)
    plt.show()



def plot_drl_compare_radar(compare_type, train_type, metric = 'mape', legend=False):
    assert compare_type in ['general', 'group','ind_user']
    assert train_type in ['general', 'group', 'ind_user', 'lopo']
    raw_dataset = pd.read_csv('../svm_model/timecare~25k_estimate_test_ind_user_encode_norm.csv')
    # raw_dataset = pd.read_csv('../rl_model/result/DRL_DDM_Main/timecare~25k_rl_estimate_test_ind_user.csv')

    general_list = ['general']
    group_list = ['none', 'random', 'rule', 'static']
    # group_list_short = ['N', 'Ra', 'Ru', 'S']
    user_list = list(np.int_(list(set(raw_dataset['user_id']))))

    if compare_type == 'general':
        angle_list = general_list
    elif compare_type == 'group':
        angle_list = group_list
    else:
        angle_list = user_list
    angle_list.sort()

    drl_ddm_dataset = table_plot('drl_ddm', train_type, compare_type, 'dict') 
    drl_only_dataset = table_plot('drl_only', train_type, compare_type, 'dict') 
    svm_dataset = table_plot('svm', train_type, compare_type, 'dict') 

    drl_ddm_dataset_new, drl_only_dataset_new, svm_dataset_new = [], [], []
    for i,each in enumerate(angle_list):
        drl_ddm_dataset_new.append(drl_ddm_dataset[each][metric])
        drl_only_dataset_new.append(drl_only_dataset[each][metric])
        svm_dataset_new.append(svm_dataset[each][metric])
    # print(drl_ddm_dataset_new)
    theta = radar_factory(len(angle_list), frame='polygon')

    fig, axs = plt.subplots(figsize=(8,5), subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    # axs.set_ylim(0.10,0.36)
    if compare_type == 'group':
        axs.set_rgrids([0.26,0.30,0.34,0.38])
        # axs.set_rgrids([0.05,0.2,0.35,0.4])
        # axs.set_rgrids([0.12,0.20,0.28,0.36])
    # else:
    #     axs.set_rgrids([0.10,0.30,0.50,0.70])
    # axs.set_title('title', weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')
    
    dataset_list = [drl_ddm_dataset_new, drl_only_dataset_new, svm_dataset_new]
    colors = ['b', 'r', 'g']
    for i,dataset in enumerate(dataset_list):
        axs.plot(theta, dataset, color=colors[i])
        axs.fill(theta, dataset, facecolor=colors[i], alpha=0.25, label='_nolegend_')
    axs.set_varlabels(angle_list)
    # axs.set_xlim(0,1)

    # add legend relative to top-left plot
    labels = ('Hybrid DRL','Pure DRL','SVM')
    if legend == True:
        legend = axs.legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small', ncol=3)

    # fig.text(0.5, 0.965, 'text', horizontalalignment='center', color='black', weight='bold', size='large')

    plt.show()



def plot_sample_rt_curve(user_id,start,end,legend = True):
    raw_dataset = pd.read_csv('../svm_model/timecare~25k_estimate_test_ind_user_encode_norm.csv')
    user_group_dict =  _get_user2group_dict(raw_dataset)
    fig, ax = plt.subplots(1, 1, figsize=(8,2), constrained_layout=True)
    print('group: ', user_group_dict[user_id])
    drl_ddm_path = '../rl_model/result/DRL_DDM_Fine_Tune/lopo_model/'+user_group_dict[user_id]+'/'+str(user_id)
    drl_only_path = '../rl_model/result/DRL_Only/lopo_model/'+user_group_dict[user_id]+'/'+str(user_id)
    all_drl_ddm_user_models = os.listdir(drl_ddm_path)
    all_drl_only_user_models = os.listdir(drl_only_path)
    all_drl_ddm_user_models_copy = all_drl_ddm_user_models.copy()
    all_drl_only_user_models_copy = all_drl_only_user_models.copy()
    for pi,sub_file in enumerate(all_drl_ddm_user_models_copy):
        if all_drl_ddm_user_models_copy[pi].split('.')[-1] == 'csv':
            del all_drl_ddm_user_models[all_drl_ddm_user_models.index(all_drl_ddm_user_models_copy[pi])]
        if all_drl_only_user_models_copy[pi].split('.')[-1] == 'csv':
            del all_drl_only_user_models[all_drl_only_user_models.index(all_drl_only_user_models_copy[pi])]
    all_drl_ddm_user_models.sort()
    all_drl_only_user_models.sort()
    print(all_drl_ddm_user_models[-1], all_drl_only_user_models[-1])
    drl_ddm_pred_data = pd.read_csv(drl_ddm_path+'/'+all_drl_ddm_user_models[-1]+'/test_pred_result.csv')
    drl_only_pred_data = pd.read_csv(drl_only_path+'/'+all_drl_only_user_models[-1]+'/test_pred_result.csv')
    assert len(drl_only_pred_data) == len(drl_ddm_pred_data)
    
    if start != None and end != None:
        drl_ddm_pred_data = drl_ddm_pred_data.loc[start:end]

    if start != None and end != None:
        drl_only_pred_data = drl_only_pred_data.loc[start:end]

    drl_ddm_pred_data['index'] = np.arange(0,len(drl_ddm_pred_data)).reshape((len(drl_ddm_pred_data),1))
    drl_only_pred_data['index'] = np.arange(0,len(drl_only_pred_data)).reshape((len(drl_only_pred_data),1))
    # sns.lineplot(x='index',y='user_resptime',data=pred_data,label='user_resptime')
    # sns.lineplot(x='index',y='est_resptime',data=pred_data,label='est_resptime')
    # sns.lineplot(x='index',y='agent_resptime',data=pred_data,label='agent_resptime')

    

    ax = sns.lineplot(x='index',y='user_resptime',data=drl_ddm_pred_data,label='Ground Truth',color='black',zorder=2,linestyle='--')
    ax = sns.lineplot(x='index',y='agent_resptime',data=drl_ddm_pred_data,label='Hybrid DRL',color='b',zorder=2)
    ax = sns.lineplot(x='index',y='agent_resptime',data=drl_only_pred_data,label='Pure DRL',color='g',zorder=2)
    ax = sns.scatterplot(x='index',y='user_resptime',data=drl_ddm_pred_data,color='black',zorder=3) # ,label='label'
    ax = sns.scatterplot(x='index',y='agent_resptime',data=drl_ddm_pred_data,color='b',zorder=3) # ,label='rl'
    ax = sns.scatterplot(x='index',y='agent_resptime',data=drl_only_pred_data,color='g',zorder=3) # ,label='rl'
    border_up = np.maximum(np.maximum(drl_ddm_pred_data['user_resptime'], drl_ddm_pred_data['agent_resptime']), drl_only_pred_data['agent_resptime'])
    border_down = np.minimum(np.minimum(drl_ddm_pred_data['user_resptime'], drl_ddm_pred_data['agent_resptime']), drl_only_pred_data['agent_resptime'])
    # ax.fill_between(drl_ddm_pred_data['index'], border_up, border_down, color='b', alpha=.5)
    
    drl_ddm_up = np.maximum(drl_ddm_pred_data['user_resptime'], drl_ddm_pred_data['agent_resptime'])
    drl_ddm_down = np.minimum(drl_ddm_pred_data['user_resptime'], drl_ddm_pred_data['agent_resptime'])
    drl_only_up = np.maximum(drl_only_pred_data['user_resptime'], drl_only_pred_data['agent_resptime'])
    drl_only_down = np.minimum(drl_only_pred_data['user_resptime'], drl_only_pred_data['agent_resptime'])
    ax.fill_between(drl_ddm_pred_data['index'], drl_ddm_up, drl_ddm_down, color='b', edgecolor='none', alpha=.5,zorder=1)
    ax.fill_between(drl_ddm_pred_data['index'], drl_only_up, drl_only_down, color='g',edgecolor='none', alpha=.5,zorder=1)
    if legend == True:
        ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1))
    else:
        ax.legend([])

    
    
    # axins1 = inset_axes(ax, width="45%", height="28.57%", bbox_to_anchor=(-0.41,-0,1,1), bbox_transform=ax.transAxes) # #, bbox_to_anchor=(1.05,0,1,1)
    # axins1.set_yticks([0,1], ['0','1'])
    # # axins1.set_xlabel('accuracy')
    # axins1.yaxis.set_ticks_position("left")
    # axins1.tick_params(direction='in')
    # axins1.spines['right'].set_visible(False)
    # axins1.spines['top'].set_visible(False)
    # axins1.spines['bottom'].set_visible(False)
    # axins1.axes.get_xaxis().set_visible(False)

    plt.show()
   

def plot_pearson_correlation():
    raw_dataset = pd.read_csv('../svm_model/timecare~25k_estimate_test_ind_user_encode_norm.csv')
    user_group_dict = _get_user_group_dict('../svm_model/timecare~25k_estimate_test_ind_user_encode_norm.csv')
    user_list = list(np.int_(list(set(raw_dataset['user_id']))))
    group_list = ['none','static','random','rule']
    general_list = ['general']
    all_list_dict = {'general': general_list, 'group': group_list, 'ind_user': user_list}
    marker_size_dict = {'general': 200, 'group': 100, 'ind_user': 40}
    marker_color_group = {'none': 'r', 'static': 'y', 'random': 'black', 'rule': 'b'}
    # 0.5,1,2,3,4,5,6,7,7.5
    fig, ax = plt.subplots(figsize=(7,5))
    ax2 = ax.axes.twinx()
    general_mape_results = []
    for ti,train_type in enumerate(['general', 'group', 'ind_user', 'lopo']):
        print('train_type: ', train_type)
        general_mape_results.append(np.around(table_plot('drl_only', train_type, 'general', 'dict')['general']['mape'],decimals=4)) 
        general_mape_results.append(np.around(table_plot('drl_ddm', train_type, 'general', 'dict')['general']['mape'],decimals=4)) 
        for ci,compare_type in enumerate(['general', 'group', 'ind_user']):
            drl_ddm_dataset = table_plot('drl_ddm', train_type, compare_type, 'dict') 
            drl_only_dataset = table_plot('drl_only', train_type, compare_type, 'dict')
            target_list = all_list_dict[compare_type]
            print('compare_type: ', compare_type)
            print('drl_ddm_dataset: ', drl_ddm_dataset)
            print('drl_only_dataset: ', drl_only_dataset)
            for i,item_id in enumerate(target_list):
                dot_color = 'gray' if compare_type != 'group' else marker_color_group[item_id]
                dot_alpha = 0.4 if compare_type == 'ind_user' else 0.8
                plt.scatter(drl_ddm_dataset[item_id]['pearson'], 2*(ti+1), s=marker_size_dict[compare_type], c=dot_color, alpha=dot_alpha, edgecolors='none')
                plt.scatter(drl_only_dataset[item_id]['pearson'], 2*(ti+1)-1, s=marker_size_dict[compare_type], c=dot_color, alpha=dot_alpha, edgecolors='none')
    plt.axvline(x=0,ymin=0,ymax=9,color='gray',alpha=0.5,linestyle='--')    
    plt.axhline(y=2.5,xmin=-0.2,xmax=1,color='gray',alpha=0.5,linestyle='--')    
    plt.axhline(y=4.5,xmin=-0.2,xmax=1,color='gray',alpha=0.5,linestyle='--')    
    plt.axhline(y=6.5,xmin=-0.2,xmax=1,color='gray',alpha=0.5,linestyle='--')    
    
    # ylabels = ('DO: General','DD: General','DO: Group','DD: Group','DO: Individual','DD: Individual','DO: LOPO','DD: LOPO')
    ylabels = ('PD: All','HD: All','PD: Group','HD: Group','PD: Ind','HD: Ind','PD: LOPO','HD: LOPO')
    ax.set_ylim(0,9)
    ax.set_yticks(np.arange(1,len(ylabels)+1))
    ax.set_yticklabels(ylabels)
    
    ylabels_2 = general_mape_results
    ax2.set_ylim(0,9)
    ax2.set_yticks(np.arange(1,len(ylabels_2)+1))
    ax2.set_yticklabels(ylabels_2)
    # plt.legend(loc='upper center',ncol=6)
    plt.show()


def explain_all(dataset_folder_path):
    group_list = ['none','static','random','rule']
    group_list.sort() # very important !!! for dataset mapping
    fig, axs = plt.subplots(2, len(group_list), figsize=(10, 6), constrained_layout=True)
    dataset_test = pd.read_csv('../svm_model/timecare~25k_estimate_test_group_encode_norm.csv')
    for gi, group in enumerate(group_list):
        dataset_test_group = dataset_test[dataset_test['group']==group]
        dataset_test_group_arr = np.array(dataset_test_group)
        folder_list = os.listdir(dataset_folder_path + '/'+ group)
        folder_list.sort()
        folder_path = dataset_folder_path + '/'+ group + '/' + folder_list[-1]
        svm_table = np.array(pd.read_csv(folder_path + '/test_pred_result.csv'))
        action_table = read_text(folder_path + '/test_pred_result_trajectory.txt')
        assert len(svm_table) == len(action_table)
        assert len(svm_table) == len(dataset_test_group_arr)
        length = len(action_table)
        for ai in range(length):
            base_trace, _, delta_proba = generate_base_trajectory(dataset_test_group_arr[ai][-1],dataset_test_group_arr[ai][-2], frequency = 5, init_interval = 20, delta = 0.001, max_time_set = 10)
            stimuli_trace = generate_stimuli_trajectory(action_table[ai], delta_proba)
            final_trace = integrate_trajectory(base_trace, stimuli_trace, stimuli_discounter = 0.01)
            axs[0][gi].plot(stimuli_trace, label = 'stimuli_trace')
            axs[0][gi].set_title(group)
    
    dataset = pd.read_csv('../dataset/timecare~25k_block_5.csv')
    dataset = dataset[(dataset['session']=='formal')&(dataset['block_id']!=1)]
    sns.boxplot(ax = axs[1][0], x = 'group', y = 'rela_1_resptime', data = dataset, palette="Greens", flierprops = dict(markerfacecolor='w', marker='o'))
    
    group_list = ['none','static','random','rule']
    dataset_test = pd.read_csv('../svm_model/timecare~25k_estimate_test_group_encode_norm.csv')
    box_x = []
    mean_data = []
    std_data = []
    slope_data = []
    for gi, group in enumerate(group_list):
        dataset_test_group = dataset_test[dataset_test['group']==group]
        dataset_test_group_arr = np.array(dataset_test_group)
        folder_list = os.listdir(dataset_folder_path + '/'+ group)
        folder_list.sort()
        folder_path = dataset_folder_path + '/'+ group + '/' + folder_list[-1]
        svm_table = np.array(pd.read_csv(folder_path + '/test_pred_result.csv'))
        action_table = read_text(folder_path + '/test_pred_result_trajectory.txt')
        assert len(svm_table) == len(action_table)
        assert len(svm_table) == len(dataset_test_group_arr)
        length = len(action_table)
        for ai in range(length):
            base_trace, _, delta_proba = generate_base_trajectory(dataset_test_group_arr[ai][-1],dataset_test_group_arr[ai][-2], frequency = 5, init_interval = 20, delta = 0.001, max_time_set = 10)
            stimuli_trace = generate_stimuli_trajectory(action_table[ai], delta_proba)
            slope_data.append((stimuli_trace[-1]-stimuli_trace[0])/len(stimuli_trace))
            std_data.append(np.std(action_table[ai]))
            mean_data.append(np.mean(stimuli_trace))
            box_x.append(group)
    box_x = np.array(box_x).reshape((len(box_x),1))
    mean_data = np.array(mean_data).reshape((len(mean_data),1))
    std_data = np.array(std_data).reshape((len(std_data),1))
    slope_data = np.array(slope_data).reshape((len(slope_data),1))
    mean_table = pd.DataFrame(mean_data, columns=['mean'])
    std_table = pd.DataFrame(std_data, columns=['std'])
    slope_table = pd.DataFrame(slope_data, columns=['slope'])
    mean_table['group'] = box_x
    std_table['group'] = box_x
    slope_table['group'] = box_x
    
    sns.boxplot(ax = axs[1][1], x = 'group', y = 'mean', data = mean_table, palette="Greens", flierprops = dict(markerfacecolor='w', marker='o'))
    sns.boxplot(ax = axs[1][2], x = 'group', y = 'std', data = std_table, palette="Greens", flierprops = dict(markerfacecolor='w', marker='o'))
    sns.boxplot(ax = axs[1][3], x = 'group', y = 'slope', data = slope_table, palette="Greens", flierprops = dict(markerfacecolor='w', marker='o'))

    plt.show()




# f2: B
plot_dataset_four_type('../dataset/timecare~25k_block_5.csv')


# f3: A,B,C,D
# dataset_path = '../dataset/timecare~25k_block_5.csv'
# plot_dataset_distribution(dataset_path,'rela_1_resptime','day','line')
# plot_dataset_distribution(dataset_path,'rela_1_resptime','day','step')
# plot_dataset_distribution(dataset_path,'rela_1_accuracy','day','line')
# plot_dataset_distribution(dataset_path,'rela_1_accuracy','day','step')
# plot_dataset_distribution(dataset_path,'rela_focus','day','line')
# plot_dataset_distribution(dataset_path,'rela_focus','day','step')
# plot_dataset_distribution(dataset_path,'rela_anxiety','day','line')
# plot_dataset_distribution(dataset_path,'rela_anxiety','day','step')

# f3: E,F,G,H
# dataset_path = '../dataset/timecare~25k_block_5.csv'
# plot_dataset_bar_seaborn(dataset_path, target_type = 'rela_1_resptime', x_type = 'group', z_type = 'block_id')
# plot_dataset_bar_seaborn(dataset_path, target_type = 'rela_1_accuracy', x_type = 'group', z_type = 'block_id')
# plot_dataset_bar_seaborn(dataset_path, target_type = 'rela_focus', x_type = 'group', z_type = 'block_id')
# plot_dataset_bar_seaborn(dataset_path, target_type = 'rela_anxiety', x_type = 'group', z_type = 'block_id')



# # f4: A
# math_folder_path = '../math_answer_agent/'
# plot_confusion_matrix(math_folder_path+'/test_numeric_out.csv')

# # f4: B
# math_folder_path = '../math_answer_agent/'
# plot_math_agent(math_folder_path)

# # f4: C
# root_dir = '../svm_model/'
# plot_svm_math_feature(root_dir+'/general_metrics_encode_norm.csv', root_dir+'/general_metrics_string_norm.csv', root_dir+'/general_metrics_norm.csv')


# f5: A, B, C, D, E, F, G, H
# plot_drl_compare_radar(compare_type = 'ind_user', train_type = 'ind_user', metric = 'mape', legend=True)
# plot_drl_compare_radar(compare_type = 'ind_user', train_type = 'group', metric = 'mape', legend=True)
# plot_drl_compare_radar(compare_type = 'ind_user', train_type = 'general', metric = 'mape', legend=True)
# plot_drl_compare_radar(compare_type = 'ind_user', train_type = 'lopo', metric = 'mape', legend=True)
# plot_drl_compare_radar(compare_type = 'group', train_type = 'ind_user', metric = 'mape', legend=True)
# plot_drl_compare_radar(compare_type = 'group', train_type = 'group', metric = 'mape', legend=True)
# plot_drl_compare_radar(compare_type = 'group', train_type = 'general', metric = 'mape', legend=True)
# plot_drl_compare_radar(compare_type = 'group', train_type = 'lopo', metric = 'mape', legend=True)


# # f6: A, B, C, D
# plot_sample_rt_curve(user_id = 13, start = 287, end = 320, legend = True) 
# plot_sample_rt_curve(user_id = 2, start = 464, end = 496, legend = False) 
# plot_sample_rt_curve(user_id = 7, start = 80, end = 113, legend = False)
# plot_sample_rt_curve(user_id = 27, start = 447, end = 478, legend = False)

# # f6: E
# plot_pearson_correlation()


# # f6: F,G
# plot_all_train_return('../rl_model/result/DRL_Only/user_model/',model_type='pure',norm=60)
# plot_all_train_return('../rl_model/result/DRL_Only/user_model/',model_type='pure')
# plot_all_train_return('../rl_model/result/DRL_DDM_Fine_0.01/user_model/',model_type='hybrid')
# plot_all_train_return('../rl_model/result/DRL_DDM_Fine_Tune/user_model/',model_type='hybrid',len_max=50000)


# f7: all subfigures (A,B,C,D,E,F,G,H)
# dataset_folder_path = '../rl_model/result/DRL_DDM_Fine_0.01/group_model/'
# explain_all(dataset_folder_path)

