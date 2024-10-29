from nni.experiment import Experiment

# 定义搜索空间
search_space = {
    "cfg_den_2": {
        "_type": "choice",
        "_value": [25, 50, 75, 100]
    },
    "num_grid": {
        "_type": "choice",
        "_value": [8, 10,  14,  18,  22]
    },
    "cfg_den_deal": {
        "_type": "choice",
        "_value": ["relu", "tanh", "sigmoid"]
    },
    "cfg_den_deal_2": {
        "_type": "choice",
        "_value": ["relu", "tanh", "sigmoid"]
    },
    "cfg_deal": {
        "_type": "choice",
        "_value": ["relu", "tanh", "sigmoid"]
    },
    "knots1": {
        "_type": "uniform",
        "_value": [0.1, 0.5]
    },
    "knots2": {
        "_type": "uniform",
        "_value": [0.5, 0.9]
    }
}

# 创建本地实验
experiment = Experiment('local')

# 设置实验相关配置
experiment.config.trial_command = 'python ./news_100_tune.py'  # 执行的训练命令
experiment.config.trial_code_directory = '.'  # 代码所在的目录

experiment.config.search_space = search_space  # 搜索空间

# 配置调优器（TPE）
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {
    'optimize_mode': 'minimize'}  # 修正：正确设置调优器参数

# 设置实验的最大试验次数和并发数
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 2

# 运行实验
experiment.run(10067)

# 停止实验
experiment.stop()
