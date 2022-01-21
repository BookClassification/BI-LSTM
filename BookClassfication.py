import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import jieba as jb
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchnlp.encoders.text import StaticTokenizerEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.optim import Adam
from itertools import repeat
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

df = pd.read_csv("dataset.csv", encoding="utf8")
df = df[['cat', 'keyword']]
print("数据总量: %d ." % len(df))
print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
print("在 keyword 列中总共有 %d 个空值." % df['keyword'].isnull().sum())
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
data_count = df['cat'].value_counts()
data_count_kw = {'cat':data_count.index, 'count':data_count}
df_cat = pd.DataFrame(data=data_count_kw).reset_index(drop=True)
print(df_cat)
df_cat.plot(x='cat', y='count', kind='bar', legend=False,  figsize=(7, 5))
plt.title(u"类目分布")
plt.ylabel(u'数量', fontsize=18)
plt.xlabel(u'类目', fontsize=18)
plt.show()
df['cat_id'] = df['cat'].factorize()[0]

cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)

cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)

print(cat_id_df)

# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 加载停用词
stopwords = stopwordslist("chineseStopWords.txt")

# 删除除字母,数字，汉字以外的所有符号
df['clean_keyword'] = df['keyword'].apply(remove_punctuation)
print(df.sample(10))

# 分词，并过滤停用词
df['cut_keyword'] = df['clean_keyword'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
print(df.head())

# 设置最频繁使用的30000个词
MAX_NB_WORDS = 30000

# 每条cut_keyword最大的长度50
MAX_SEQUENCE_LENGTH = 50

# 设置Embeddingceng层的维度
EMBEDDING_DIM = 128

tok = StaticTokenizerEncoder(sample=df['cut_keyword'].values, tokenize=lambda x: x.split(),reserved_tokens=['<pad>']) # 初始化标注器

word_index = tok.token_to_index  # 查看对应的单词和数字的映射关系dict

print(word_index)

X = [tok.encode(text) for text in df['cut_keyword'].values] # 通过texts_to_sequences 这个dict可以将每个string的每个词转成数字

X.append(torch.tensor([0 for i in range(MAX_SEQUENCE_LENGTH)]))

# print(X)

# print(pd.DataFrame(X))
#在前填充 使得矩阵前部分为0
tempX=[]
for i in X:
    t=F.pad(i,(MAX_SEQUENCE_LENGTH-len(i),0))
    tempX.append(t)

#最后一行数据都是0 直接删除
tempX.pop()

#生成张量
X=torch.cat(tempX,0).view(-1,MAX_SEQUENCE_LENGTH)
# X = pad_sequence(X).T[:]

print(X,X.shape)

Y = df['cat_id'].values

print(type(Y),Y.shape)
print(type(X),X.shape)

X = np.array([[int(j) for j in i] for i in X])
Y = np.array(Y)

print(type(Y),Y.shape)
print(type(X),X.shape)

X=torch.from_numpy(X)
Y=torch.from_numpy(Y)

print(type(Y),Y.shape)
print(type(X),X.shape)
# 拆分训练集和测试集

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=True)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)


class pyt_SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, drop=0.5):
        super(pyt_SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape

        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)

class LSTMnet(nn.Module):
    def __init__(self, output_size, hidden_dim,embedding_dim, bidirectional):
        super(LSTMnet, self).__init__()
        self.output_size = output_size
        self.liner1_input_size=hidden_dim*2 if bidirectional else hidden_dim
        self.Embedding = nn.Embedding(MAX_NB_WORDS, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=0.2, bidirectional=bidirectional,num_layers=2)
        self.dropout1 = nn.Dropout(0.25)
        self.linear1 = nn.Linear(self.liner1_input_size, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(64, output_size)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x=self.Embedding(x)
        lstm_out,hidden = self.lstm(x)
        # print(lstm_out.shape)
        lstm_out=lstm_out[:,-1] #取最后一步输出
        # print(lstm_out.shape)
        out=self.dropout1(lstm_out)
        out=self.linear1(out)
        # print(out.shape)
        out=self.dropout2(out)
        out=self.linear2(out)
        # print(out.shape)
        out=self.sig(out)
        return out


# 训练模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 20

batch_size = 1200

now_time = time.time()

model = LSTMnet(output_size=5,
                hidden_dim=50,
                embedding_dim=EMBEDDING_DIM,
                bidirectional=True)

loss_function = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(),
                 lr=0.01,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False)

reduce_lr = ReduceLROnPlateau(optimizer,
                              mode='min',
                              factor=0.2,
                              patience=5,
                              min_lr=0.001)
# X_train=X_train.t()
#通过dataloader实现分批次训练
train_dataset = TensorDataset(X_train,Y_train)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset = TensorDataset(X_test,Y_test)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1200,shuffle=True)

#记录用于绘图
Train_loss=[]
Train_accu=[]
Test_loss =[]
Test_accu =[]

#到device上训练
model.to(device)
loss_function.to(device)

for epoch in range(epochs):
    model.train()
    train_loss=0
    correct=0
    for i,data in enumerate(train_dataloader):
        inputs,labels=data
        inputs,labels= Variable(inputs),Variable(labels)
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        # print(outputs.shape,labels.shape)
        optimizer.zero_grad()
        loss=loss_function(outputs,labels)
        loss.backward()
        #如果创建了lr_scheduler对象之后，先调用scheduler.step()，再调用optimizer.step()，则会跳过了第一个学习率的值。
        # 调用顺序
        # loss.backward()
        # optimizer.step()
        # scheduler.step()
        optimizer.step()
        reduce_lr.step(loss)
        _,preds=outputs.max(1)
        correct=preds.eq(labels).sum().item()/len(labels)
        train_loss+=loss.item()

    Train_loss.append(train_loss)
    Train_accu.append(100 * correct)
    print("Epoch : {} ,Train_loss : {:.6f}".format(epoch, train_loss))
    print('Train_Accuracy : {:.6f}'.format(100 * correct))

    model.eval()
    with torch.no_grad():
        test_accu=0
        test_loss=0
        for i,data in enumerate(test_dataloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # print(outputs.shape,labels.shape)
            loss = loss_function(outputs, labels)
            _, preds = outputs.max(1)
            correct = preds.eq(labels).sum().item() / len(labels)
            test_loss += loss.item()
        Test_loss.append(test_loss)
        Test_accu.append(100*correct)
        print("Epoch : {} ,Test_loss : {:.6f}".format(epoch, test_loss))
        print('Test_Accuracy : {:.6f}'.format(100 * correct))


total_time = time.time() - now_time
print("total time is: ", total_time)
#绘制训练结果图像
plt.title('Train_process')
plt.plot(range(epochs),Train_loss, label='train_loss',color='b')
plt.plot(range(epochs),Train_accu, label='train_accu',color='r')
plt.legend()
plt.show()
#绘制测试结果图像
plt.title('Test_process')
plt.plot(range(epochs), Test_loss, label='test_loss',color='b')
plt.plot(range(epochs), Test_accu, label='test_accu',color='r')
plt.legend()
plt.show()
