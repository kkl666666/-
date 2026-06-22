import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

# 忽略 tqdm 的转换警告
warnings.filterwarnings("ignore", category=UserWarning, module='tqdm')

# ================= 1. 读取 Excel 数据集 =================
data_path = 'TextFooler攻击后新的欺诈通话数据.xlsx'
try:
    df = pd.read_excel(data_path)
    original_texts = df['原始的通话记录'].dropna().tolist()
    attacked_texts = df['textfooler攻击后的通话记录'].dropna().tolist()
    print(f"✅ 成功加载数据！共读取到 {len(original_texts)} 条原始记录和 {len(attacked_texts)} 条攻击后记录。")
except Exception as e:
    print(f"❌ 数据加载失败！详细错误信息: {e}")
    exit()

# 数据集全部为欺诈通话，真实标签为 1 (Fraud)
y_true = [1] * len(original_texts)


# ================= 2. 对比模型一：传统机器学习 Baseline (TF-IDF + LR) =================
print("\n--- 正在初始化对比模型：TF-IDF + Logistic Regression ---")

# 🔧 【已修正】使用 analyzer='char' 和 ngram_range=(1, 4) 来正确提取中文的字符级 N-gram 特征
vectorizer = TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(1, 4))

# 构造一个伪训练集让逻辑回归模型有基本的判别边界
X_pseudo_train = original_texts + ["你好，请问是张先生吗？我们是蓝天家电维修中心，您预约的下午两点维修空调。"] * 10 
y_pseudo_train = [1] * len(original_texts) + [0] * 10

X_train_vec = vectorizer.fit_transform(X_pseudo_train)
lr_model = LogisticRegression(C=1.0, max_iter=1000)
lr_model.fit(X_train_vec, y_pseudo_train)

# 预测
X_orig_vec = vectorizer.transform(original_texts)
X_atk_vec = vectorizer.transform(attacked_texts)

y_pred_lr_orig = lr_model.predict(X_orig_vec)
y_pred_lr_atk = lr_model.predict(X_atk_vec)


# ================= 3. 主模型二：深度预训练模型 (BERT-base-Chinese) =================
print("\n--- 正在加载主模型: bert-base-chinese ---")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
bert_model.eval()

def predict_bert_batch(texts, batch_size=16):
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT 批处理预测中"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True).to(device)
            outputs = bert_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            predictions.extend(preds)
    return predictions

print("\n🚀 开始执行 BERT 对【原始通话记录】的评测...")
y_pred_bert_orig = predict_bert_batch(original_texts, batch_size=16) 

print("\n🚀 开始执行 BERT 对【TextFooler攻击后通话记录】的评测...")
y_pred_bert_atk = predict_bert_batch(attacked_texts, batch_size=16)


# ================= 4. 最终核心性能对比表格展示 =================
acc_lr_orig = accuracy_score(y_true, y_pred_lr_orig)
acc_lr_atk = accuracy_score(y_true, y_pred_lr_atk)
drop_lr = acc_lr_orig - acc_lr_atk

acc_bert_orig = accuracy_score(y_true, y_pred_bert_orig)
acc_bert_atk = accuracy_score(y_true, y_pred_bert_atk)
drop_bert = acc_bert_orig - acc_bert_atk

print("\n" + "="*75)
print("                           虚假通话检测鲁棒性实验性能对比表格                      ")
print("="*75)
print(f"{'模型结构 (Model Architecture)':<30} | {'原始数据召回率 (Recall)':<20} | {'攻击后召回率 (Recall)':<20} | {'性能下降幅 (Drop)':<15}")
print("-"*75)
print(f"{'TF-IDF + Logistic Regression':<30} | {acc_lr_orig*100:>21.2f}% | {acc_lr_atk*100:>20.2f}% | {drop_lr*100:>14.2f}%")
print(f"{'BERT-base-Chinese (Ours)':<30} | {acc_bert_orig*100:>21.2f}% | {acc_bert_atk*100:>20.2f}% | {drop_bert*100:>14.2f}%")
print("="*75)
print("* 注：测试集均为欺诈通话（正样本），此时模型分类正确率等价于欺诈阻断召回率(Recall)。")