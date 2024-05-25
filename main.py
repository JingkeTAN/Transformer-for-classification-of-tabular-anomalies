import torch
from torch.utils.data import TensorDataset, DataLoader, random_split,Dataset
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from torchmetrics import AUROC
import datetime
from tqdm import tqdm
from copy import deepcopy
import sys
#数据预处理
# 读取 Excel 表格
df = pd.read_csv(r'Hong Kong-Zhuhai-Macal\gza2022-11-20 to 2022-11-20.csv')

# 使用 drop 方法删除不需要的列
df = df.drop([df.columns[0],'License','StakePosition','ObjectClass'], axis=1)
#修正N_Point列
df['GlobalID_Count'] = df.groupby('GlobalID')['GlobalID'].transform('size')
df['N_Point'] = df.groupby('GlobalID')['N_Point'].transform('size')
df = df.drop(['GlobalID_Count'], axis=1)

# 把轨迹长度小于47的去掉
count_less_than_47 = (df['N_Point'] < 47).sum()
df = df[df['N_Point'] >= 47]
#删除小于100条的异常状态
# 获取每个唯一值的计数
abnormal_state_counts = df['AbnormalState'].value_counts()

# 找到数量小于 100 的唯一值
values_to_remove = abnormal_state_counts[abnormal_state_counts < 100].index

# 使用布尔索引删除这些行
df = df[~df['AbnormalState'].isin(values_to_remove)]

# 使用布尔索引过滤掉 AbnormalState 值为 68 和 69 的行
df = df[(df['AbnormalState'] != 68) & (df['AbnormalState'] != 69)]


'''
车辆状态（0-正常，1-蛇形驾驶，2-超速，3-蛇形驾驶+超速，4-低
速，8-急加速急减速，10急加速急减速+超速）
上面的部分清洗了无用的标签和过短的轨迹
'''
#保留标签
label = pd.DataFrame()
label['AbnormalState'] = df['AbnormalState']
'''因为打算保留标签的多样性，所以下面两行没用了'''
label['AbnormalState'] = label['AbnormalState'].apply(lambda x: 1 if x != 0 else x)
label['AbnormalState'] = label['AbnormalState'].apply(lambda x: 0 if x == 0 else x)
columns = ['Timestamp','GlobalID', 'N_Point','VelocityX','VelocityY','IdxPoint', 'PositionX' ,'PositionY','ActualLatitudeRelativeBias']
df = df[columns]
# 将 GlobalID 列的值按顺序映射为 [1, 2, ...,516]
df['GlobalID'] = df.groupby('GlobalID').ngroup() + 1
df[['PositionX', 'ActualLatitudeRelativeBias', 'VelocityY']] = \
df[['PositionX', 'ActualLatitudeRelativeBias', 'VelocityY']].apply(lambda x: 1000*(x - x.min()) / (x.max() - x.min())).round(0)
df = df.abs()
# 假设您的数据框名为df_normalized，其中X表示特征，y表示目标变量
X = df  # 特征
y = label  # 目标变量
# 创建分层抽样对象
sss_train = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
sss_temp = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=42)

# 使用分层抽样拆分数据，使得每个集的分类比例相同
train_index, temp_index = next(sss_train.split(X, y))
X_train, X_temp = X.iloc[train_index], X.iloc[temp_index]
y_train, y_temp = y.iloc[train_index], y.iloc[temp_index]

val_index, test_index = next(sss_temp.split(X_temp, y_temp))
X_val, X_test = X_temp.iloc[val_index], X_temp.iloc[test_index]
y_val, y_test = y_temp.iloc[val_index], y_temp.iloc[test_index]
# (139719, 7) (27943, 7) (65203, 7) (139719, 1) (27943, 1) (65203, 1)

# 将特征分成两组
# 获取特定列的整数索引
cols = ['GlobalID', 'N_Point', 'PositionY']
#first_input_indices = [X_train.columns.get_loc(col) for col in cols]

# 选择第一个输入的列
X_train_f = X_train.iloc[:, [X_train.columns.get_loc(col) for col in cols]]
X_val_f = X_val.iloc[:, [X_val.columns.get_loc(col) for col in cols]]
X_test_f = X_test.iloc[:, [X_test.columns.get_loc(col) for col in cols]]
# 选择第二个输入的列，即除了指定列之外的所有列
X_train_s = X_train.drop(cols, axis=1)
X_val_s = X_val.drop(cols, axis=1)
X_test_s = X_test.drop(cols, axis=1)

#将dataframe变成array,方便后面进行tensor转换
X_train_f,X_train_s,X_val_f,X_val_s,X_test_f,X_test_s,y_train,y_val,y_test =\
X_train_f.values,X_train_s.values,X_val_f.values,X_val_s.values,X_test_f.values,X_test_s.values,y_train.values,y_val.values,y_test.values



# 创建TensorDatasets
train_dataset = TensorDataset(torch.tensor(X_train_f).long(), torch.tensor(X_train_s).long(), torch.tensor(y_train).long())
val_dataset = TensorDataset(torch.tensor(X_val_f).long(), torch.tensor(X_val_s).long(), torch.tensor(y_val).long())
test_dataset = TensorDataset(torch.tensor(X_test_f).long(), torch.tensor(X_test_s).long(), torch.tensor(y_test).long())

# 创建DataLoaders
dl_train = DataLoader(train_dataset, shuffle=True, batch_size=800)
dl_val = DataLoader(val_dataset, shuffle=False, batch_size=800)
dl_test = DataLoader(test_dataset, shuffle=False, batch_size=800)




















net = FTTransformer(
    categories = (466, 283, 4),      # GlobalID\N_Point\PositionY
    num_continuous = 6,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1 ,                   # feed forward dropout
    num_special_tokens = 2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
loss_fn = nn.BCEWithLogitsLoss()
loss_fn.to(device)
torch.manual_seed(42)




















# 定义一个打印日志信息的函数
def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")


# 使用二元交叉熵损失函数创建损失函数对象,自带sigmoid，据说比在前面【定义创建神经网络的函数】自己加效果好
# loss_fn = nn.BCEWithLogitsLoss()
# 使用Adam优化器进行模型参数的优化，学习率为0.01
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 定义一个字典，用于存储模型评估指标，这里包括准确度
metrics_dict = {"auc": AUROC(task="binary")}

# 训练的总轮数
epochs = 100

# 定义用于保存最佳模型权重的文件路径
ckpt_path = r'C:\Users\Kobayakawa\Desktop\模型储存\sequential\ftttran.pt'

# Early Stopping 相关设置
monitor = "val_cusloss"  # 用于监控模型性能的指标
patience = 10  # 当连续多少轮性能没有提升时，触发早停
mode = "min"  # 监控指标的模式，"max"表示监控指标越大越好

# 存储训练历史信息的字典
history = {}

# 开始训练循环
for epoch in range(1, epochs + 1):
    printlog("Epoch {0} / {1}".format(epoch, epochs))

    # 1，训练阶段 -------------------------------------------------
    net.train()  # 设置模型为训练模式

    total_loss, step = 0, 0

    # 使用tqdm库显示训练进度，并设置文件输出为sys.stdout
    loop = tqdm(enumerate(dl_train), total=len(dl_train), file=sys.stdout)
    train_metrics_dict = deepcopy(metrics_dict)  # 复制评估指标字典，用于存储本轮训练的指标值

    for i, batch in loop:

        (features_f, features_s, labels) = batch
        features_f = features_f.to(device)
        features_s = features_s.to(device)
        labels = labels.to(device)
        # 前向传播
        preds = net(features_f,features_s)
        preds = preds.float()
        labels = labels.float()
        # 计算误报率和精确率


        loss = loss_fn(preds, labels)
        # 计算损失

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 计算指标
        step_metrics = {"train_" + name: metric_fn(preds, labels).item()
                        for name, metric_fn in train_metrics_dict.items()}

        # 创建包含本步骤的训练损失和指标的字典
        step_log = dict({"train_loss": loss.item()}, **step_metrics)

        total_loss += loss.item()

        step += 1
        if i != len(dl_train) - 1:
            loop.set_postfix(**step_log)
        else:
            # 如果是本轮的最后一个批次，计算本轮的平均训练损失
            epoch_loss = total_loss / step

            # 计算并记录本轮的平均训练指标
            epoch_metrics = {"train_" + name: metric_fn.compute().item()
                             for name, metric_fn in train_metrics_dict.items()}

            # 创建包含本轮的平均训练损失和指标的字典
            epoch_log = dict({"train_loss": epoch_loss}, **epoch_metrics)

            # 更新进度条的显示，显示本轮的平均训练损失和指标
            loop.set_postfix(**epoch_log)

            # 重置本轮的训练指标，以便下一轮使用
            for name, metric_fn in train_metrics_dict.items():
                metric_fn.reset()

    # 将本轮的训练损失和指标记录到训练历史中
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]

    # 2，验证阶段 -------------------------------------------------
    net.eval()  # 设置模型为评估模式

    total_loss, step = 0, 0
    loop = tqdm(enumerate(dl_val), total=len(dl_val), file=sys.stdout)

    val_metrics_dict = deepcopy(metrics_dict)  # 复制评估指标字典，用于存储本轮验证的指标值

    with torch.no_grad():  # 验证阶段是不计算梯度的
        for i, batch in loop:
            (features_f, features_s, labels) = batch
            features_f = features_f.to(device)
            features_s = features_s.to(device)
            labels = labels.to(device)
            # 前向传播
            preds = net(features_f, features_s)
            preds = preds.float()
            labels = labels.float()
            # 计算误报率和精确率




            loss = loss_fn(preds, labels)


            # 计算指标
            step_metrics = {"val_" + name: metric_fn(preds, labels).item()
                            for name, metric_fn in val_metrics_dict.items()}

            # 创建包含本步骤的验证损失和指标的字典
            step_log = dict({"val_cusloss": loss.item()}, **step_metrics)

            total_loss += loss.item()
            step += 1
            if i != len(dl_val) - 1:
                loop.set_postfix(**step_log)
            else:
                # 如果是本轮的最后一个批次，计算本轮的平均验证损失
                epoch_loss = (total_loss / step)

                # 计算并记录本轮的平均验证指标
                epoch_metrics = {"val_" + name: metric_fn.compute().item()
                                 for name, metric_fn in val_metrics_dict.items()}

                # 创建包含本轮的平均验证损失和指标的字典
                epoch_log = dict({"val_cusloss": epoch_loss}, **epoch_metrics)

                # 更新进度条的显示，显示本轮的平均验证损失和指标
                loop.set_postfix(**epoch_log)

                # 重置本轮的验证指标，以便下一轮使用
                for name, metric_fn in val_metrics_dict.items():
                    metric_fn.reset()

    # 将本轮的验证损失和指标记录到训练历史中
    epoch_log["epoch"] = epoch
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]

    # 3，Early Stopping -------------------------------------------------
    arr_scores = history[monitor]  # 获取历史上的监控指标数值
    best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)

    # 如果当前的模型性能比历史上的最佳性能好，保存当前模型权重
    if best_score_idx == len(arr_scores) - 1:
        torch.save(net.state_dict(), ckpt_path)
        print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                          arr_scores[best_score_idx]), file=sys.stderr)

    # 如果连续多轮性能没有提升，触发早停
    if len(arr_scores) - best_score_idx > patience:
        print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
            monitor, patience), file=sys.stderr)
        break

    # 恢复历史上的最佳模型权重
    net.load_state_dict(torch.load(ckpt_path))

# 将训练历史信息转为DataFrame格式
dfhistory = pd.DataFrame(history)