import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
from modelscope.hub.snapshot_download import snapshot_download

warnings.filterwarnings("ignore", category=UserWarning, module='tqdm')
# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 读取 Excel 数据集 =================
data_path = 'TextFooler攻击后新的欺诈通话数据.xlsx'
try:
    df = pd.read_excel(data_path)
    original_texts = df['原始的通话记录'].dropna().tolist()
    attacked_texts = df['textfooler攻击后的通话记录'].dropna().tolist()
    print(f"✅ 成功加载数据！共 {len(original_texts)} 条记录。")
except Exception as e:
    print(f"❌ 数据加载失败: {e}"); exit()

y_true = [1] * len(original_texts)

# 💡 [新增模拟] 为满足任务3，依据关键词简单模拟三种策略切片 (实际应用中可用LLM打标更精准)
strategies = []
for text in original_texts:
    if any(w in text for w in ['客服', '经理', '官方', '工号', '中心']):
        strategies.append('信任建立')
    elif any(w in text for w in ['立即', '赶快', '限制', '严重', '冻结', '黑名单']):
        strategies.append('紧迫感')
    else:
        strategies.append('情感操纵')
df['strategy'] = strategies

# ================= 2. TF-IDF + LR Baseline =================
print("\n--- 正在初始化：TF-IDF + Logistic Regression ---")
vectorizer = TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(1, 4))
X_pseudo_train = original_texts + ["你好，请问是张先生吗？我们是正常客服。"] * 10 
y_pseudo_train = [1] * len(original_texts) + [0] * 10
X_train_vec = vectorizer.fit_transform(X_pseudo_train)
lr_model = LogisticRegression(C=1.0, max_iter=1000).fit(X_train_vec, y_pseudo_train)

y_pred_lr_orig = lr_model.predict(vectorizer.transform(original_texts))
y_pred_lr_atk = lr_model.predict(vectorizer.transform(attacked_texts))

# ================= 3. BERT-base-Chinese (带概率输出) =================
model_dir = snapshot_download('tiansz/bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained(model_dir)
bert_model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2, ignore_mismatched_sizes=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device).eval()

def predict_bert_with_prob(texts, batch_size=16):
    preds, probs = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT 预测中"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            outputs = bert_model(**inputs)
            
            # 💡 [新增] 通过 Softmax 提取预测为欺诈（Label 1）的置信度概率
            softmax_probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            
            preds.extend(batch_preds)
            probs.extend(softmax_probs)
    return preds, probs

print("\n🚀 评估原始数据...")
y_pred_bert_orig, probs_bert_orig = predict_bert_with_prob(original_texts)
print("\n🚀 评估攻击后数据...")
y_pred_bert_atk, probs_bert_atk = predict_bert_with_prob(attacked_texts)

df['bert_orig_correct'] = (np.array(y_pred_bert_orig) == 1).astype(int)
df['bert_atk_correct'] = (np.array(y_pred_bert_atk) == 1).astype(int)

# ================= 4. 绘图 1：不同诱导策略下的性能对比 (任务3核心) =================
strategy_perf = df.groupby('strategy')[['bert_orig_correct', 'bert_atk_correct']].mean()

plt.figure(figsize=(10, 6))
x = np.arange(len(strategy_perf.index))
width = 0.35
plt.bar(x - width/2, strategy_perf['bert_orig_correct']*100, width, label='原始数据召回率', color='#3A6BB1')
plt.bar(x + width/2, strategy_perf['bert_atk_correct']*100, width, label='攻击后召回率', color='#E67E22')
plt.xticks(x, strategy_perf.index, fontsize=12)
plt.ylabel('召回率 / 检测率 (%)', fontsize=12)
plt.title('图 1：不同心理诱导策略对 BERT 模型鲁棒性的影响分析', fontsize=14, fontweight='bold')
plt.ylim(0, 110)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig('strategy_robustness_comparison.png', dpi=300)
plt.show()

# ================= 5. 绘图 2：攻击前后 BERT 预测置信度分布变化 =================
plt.figure(figsize=(10, 5))
sns.kdeplot(probs_bert_orig, shade=True, color="#2ECC71", label="原始文本置信度 (Confidence)", bw_adjust=0.5)
sns.kdeplot(probs_bert_atk, shade=True, color="#E74C3C", label="攻击后文本置信度 (Confidence)", bw_adjust=0.5)
plt.title('图 2：TextFooler 语义改写攻击前后模型预测置信度分布位移图', fontsize=14, fontweight='bold')
plt.xlabel('模型判别为虚假通话的概率 (Probability Score)', fontsize=12)
plt.ylabel('密度 (Density)', fontsize=12)
plt.xlim(0, 1)
plt.legend()
plt.grid(linestyle='--', alpha=0.3)
plt.savefig('confidence_distribution_shift.png', dpi=300)
plt.show()

# ================= 6. 打印输出策略分析报表 =================
print("\n" + "="*60)
print("             各心理诱导策略下 BERT 防御性能精细化分析            ")
print("="*60)
for strat in strategy_perf.index:
    orig_acc = strategy_perf.loc[strat, 'bert_orig_correct'] * 100
    atk_acc = strategy_perf.loc[strat, 'bert_atk_correct'] * 100
    drop = orig_acc - atk_acc
    print(f"策略【{strat:<4}】-> 原始召回率: {orig_acc:>6.2f}% | 攻击后召回率: {atk_acc:>6.2f}% | 性能跌幅: {drop:>5.2f}%")
print("="*60)
