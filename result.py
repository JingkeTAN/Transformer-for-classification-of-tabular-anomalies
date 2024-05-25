import torch
from tab_transformer_pytorch import FTTransformer
from torch.utils.data import TensorDataset, DataLoader, random_split,Dataset
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from torchmetrics import AUROC
import datetime
from tqdm import tqdm
from copy import deepcopy
import sys
from torchkeras import summary




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
ckpt_path = r'C:\Users\Kobayakawa\Desktop\模型储存\sequential\ftttran.pt'
net.load_state_dict(torch.load(ckpt_path))
net.to('cpu')
# 使用神经网络进行推断，计算前10个样本的预测概率
# 使用 torch.sigmoid 函数将模型输出的 logits 转换为概率值

# y_pred_probs = torch.sigmoid(net(torch.tensor(X_test_f).long(), torch.tensor(X_test_s).long())).data
y_pred_probs = net(torch.tensor(X_test_f).long(), torch.tensor(X_test_s).long()).data
# 输出预测概率值
print(y_pred_probs)

# 预测类别
# 如果预测的概率值大于0.5，则将类别设置为1，否则设置为0
#
threshold = -0.5326346
# 使用 torch.where 函数进行类别的判定
y_pred = torch.where(
    y_pred_probs > threshold,  # 预测概率值大于0.5时，为正类别
    torch.ones_like(y_pred_probs),  # 正类别标签（1）
    torch.zeros_like(y_pred_probs)  # 负类别标签（0）
)

# 输出预测的类别
print(y_pred.shape)
print(y_pred)

'''# 假设y_pred_probs是你的模型预测概率数组
unique_probs = np.unique(y_pred_probs)
num_thresholds = len(unique_probs) - 1

print('阈值的数量:', num_thresholds) #64911

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
# 将tensor转换为numpy数组
y_pred_probs = y_pred_probs.numpy()[:600]
y_test = y_test[:600]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)
print('-------------------------------------')
# 计算F1分数
f1_scores = [f1_score(y_test, y_pred_probs > t) for t in thresholds]
# 找到最大F1分数对应的阈值
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print('最佳阈值:', optimal_threshold) #-0.85234237  -0.5326346'''




import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import itertools

# 绘制混淆矩阵
"""cm：混淆矩阵的数据。
title：图表的标题。
classes：类别的标签，默认为 ['abnormal', 'normal']。
cmap：热力图的颜色映射，默认为蓝色（plt.cm.Blues）。
save：一个布尔值，如果为 True，则保存图像。
saveas：保存图像的文件名。"""


def plot_confusion_matrix(cm, title, classes=['normal', 'abnormal'],
                          cmap=plt.cm.Blues, save=False, saveas="MyFigure.png"):
    # 用蓝色渐变色打印混淆矩阵

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 每行的某值除以某一行的总和

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1%'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save:
        plt.savefig(saveas, dpi=100)


#######测试数据的检测效果71.9
#######添加dropout后测试数据的检测auc
print(classification_report(y_test, y_pred, target_names=['normal', 'abnormal']))  # (这两个要是binary)
print("AUC: ", "{:.1%}".format(roc_auc_score(y_test, y_pred_probs)))  # 这个直接输入模型输出就行
cm = confusion_matrix(y_test, y_pred)  # (这两个要是binary)
plot_confusion_matrix(cm, title="SE Confusion Matrix - df",save=True, saveas=r"C:\Users\Kobayakawa\Desktop\模型储存\MyFigure.png")


from sklearn.metrics import roc_auc_score, confusion_matrix

# 计算AUC
auc = roc_auc_score(y_test, y_pred_probs)
print("AUC:", auc)

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 计算误报率（FPR）
fpr = fp / (fp + tn)
recall = tp / (tp + fn)
print("FPR:", fpr)
print("recall:", recall)