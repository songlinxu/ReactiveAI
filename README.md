# ReactiveAI: Modelling human logical reasoning process in dynamic environmental stress with cognitive agents

## Note

- Please do not forget to install the necessary libraries to run each file.
- We recommend to use python virtual environments like venv.

## Steps to replicate figures in the paper

- Download RL model weights from OSF (https://osf.io/nrxy4/), rename the downloaded folder as "results" and put it into the "rl_model" folder in this Github repo
- Double-check one example folder path if you make the correct path: ReactiveAI-main/rl_model/results/DRL_DDM_Fine_0.01.
- Go to "ReactiveAI-main/paper_plot" folder and make it your current path
- Run python figure_plot.py to draw all figures
- You could comment main codes to draw different figures in figure_plot.py. More details are in this file. 


If you want to train the models from beginning using our datasets, here are the steps. Note that trainning from beginning usually takes lots of time (so you'd better have CUDA) and you have to first configure the virtual environments.

## Steps to train the model from beginning
- Run the first step in our cognitive agent framework: LSTM-based logical reasoning agent trainning: go to "math_answer_agent" folder and run lstm_math.py (more comments and details are in this file).
- Run the second step in our cognitive agent framework: Transfer features from logical reasoning agents to humans' responses: go to "svm_model" folder and run main.py (more comments and details are in this file).
- Run the third/fourth step in our cognitive agent framework: DDM + DRL agents to simulate perturbation of dynamic stimuli on human logical reasoning process: go to "rl_model" folder. env_run.py is used to run DDM + DRL agents and env_run_whole.py is used to run pure DRL agents as our baseline. More details and comments are in the files.


## Related paper

https://arxiv.org/abs/2301.06216
