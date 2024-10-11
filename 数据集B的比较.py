import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AutoTokenizer, \
    AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 定义文件路径
data_path = '数据集B/'  # 数据集文件夹路径
train_file = os.path.join(data_path, 'train_data.json')  # 训练集文件路径
dev_file = os.path.join(data_path, 'dev_data.json')  # 开发集文件路径
test_file = os.path.join(data_path, 'test_data.json')  # 测试集文件路径


# 加载JSON数据集，逐行读取
def load_data_from_json(file_path):
    sentences = []  # 存储文本句子的列表
    spo_lists = []  # 存储对应的三元组列表
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 跳过空行
                item = json.loads(line)  # 将每行JSON格式数据转换为字典
                text = item['text']  # 提取文本内容
                spo_list = item['spo_list']  # 提取三元组列表
                sentences.append(text)  # 将文本添加到句子列表中
                spo_lists.append(spo_list)  # 将三元组列表添加到 spo_list 列表中
    return sentences, spo_lists  # 返回句子和三元组的列表


# 标签生成：BIO标注，用于实体识别
def generate_bio_labels(sentence, spo_list, tokenizer, label_to_id, max_len):
    tokens = tokenizer.tokenize(sentence)
    label_ids = ['O'] * len(tokens)

    def find_sublist_positions(sublist, full_list):
        """找到子列表(sublist)在完整列表(full_list)中的所有起始位置。"""
        sublist_len = len(sublist)
        for i in range(len(full_list)):
            if full_list[i:i + sublist_len] == sublist:
                return i
        return -1

    # 检查 spo_list 的类型
    if isinstance(spo_list, dict):
        spo_list = [spo_list]  # 如果是单个字典，转为列表

    # 遍历每个三元组，生成对应的 BIO 标签
    for spo in spo_list:
        subject_tokens = tokenizer.tokenize(spo['subject'])
        object_tokens = tokenizer.tokenize(spo['object'])

        # 找到subject和object在句子中的位置
        subject_pos = find_sublist_positions(subject_tokens, tokens)
        object_pos = find_sublist_positions(object_tokens, tokens)

        # 标注subject的BIO标签
        if subject_pos != -1:
            label_ids[subject_pos] = f'B-{spo["subject_type"]}'
            for i in range(subject_pos + 1, subject_pos + len(subject_tokens)):
                label_ids[i] = f'I-{spo["subject_type"]}'

        # 标注object的BIO标签
        if object_pos != -1:
            label_ids[object_pos] = f'B-{spo["object_type"]}'
            for i in range(object_pos + 1, object_pos + len(object_tokens)):
                label_ids[i] = f'I-{spo["object_type"]}'

    # 截断或填充标签
    if len(label_ids) > max_len:
        label_ids = label_ids[:max_len]
    else:
        label_ids += ['O'] * (max_len - len(label_ids))

    # 将标签转换为ID
    try:
        label_ids = [label_to_id[label] for label in label_ids]
    except KeyError as e:
        print(f"Error: {e}. Available labels: {label_to_id.keys()}")
        raise

    return label_ids


# 加载数据
train_sentences, train_spo_lists = load_data_from_json(train_file)  # 加载训练数据
dev_sentences, dev_spo_lists = load_data_from_json(dev_file)  # 加载开发数据
test_sentences, test_spo_lists = load_data_from_json(test_file)  # 加载测试数据

# 生成标签集合，并添加‘O’标签，遍历所有数据集
entity_types = set()
for spo_lists in [train_spo_lists, dev_spo_lists, test_spo_lists]:
    for spo_list in spo_lists:
        if isinstance(spo_list, dict):
            spo_list = [spo_list]
        for spo in spo_list:
            entity_types.add(spo['subject_type'])
            entity_types.add(spo['object_type'])

# 创建 BIO 标签
entity_labels = ['O'] + [f'B-{et}' for et in entity_types] + [f'I-{et}' for et in entity_types]

# 映射标签到ID
label_to_id = {label: i for i, label in enumerate(entity_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

# 打印生成的标签集合
print(f"Generated labels: {entity_labels}")

# 定义模型A和模型B
# 模型A (AnchiBERT)
config_a = BertConfig.from_pretrained('./AnchiBERT', num_labels=len(label_to_id))
tokenizer_a = BertTokenizer.from_pretrained('./AnchiBERT')
model_a = BertForTokenClassification.from_pretrained('./AnchiBERT', config=config_a)

# 模型B (GuwenBERT)
config_b = BertConfig.from_pretrained('./guwenbert-base', num_labels=len(label_to_id))
tokenizer_b = BertTokenizer.from_pretrained('./guwenbert-base')
model_b = BertForTokenClassification.from_pretrained('./guwenbert-base',config=config_b)

# 使用Sikuroberta的预训练模型和分词器（第三个模型）
config_c= BertConfig.from_pretrained('./Sikuroberta', num_labels=len(label_to_id))
tokenizer_c= BertTokenizer.from_pretrained('./Sikuroberta')
model_c= BertForTokenClassification.from_pretrained('./Sikuroberta', config=config_c)

# 使用Bert-ancient-Chinese模型的预训练模型和分词器（第四个模型）
config_d = BertConfig.from_pretrained('./Bert-ancient-Chinese', num_labels=len(label_to_id))
tokenizer_d = BertTokenizer.from_pretrained('./Bert-ancient-Chinese')
model_d = BertForTokenClassification.from_pretrained('./Bert-ancient-Chinese', config=config_d)

# 自定义数据集类，用于关系抽取任务
class RelationExtractionDataset(Dataset):
    def __init__(self, sentences, spo_lists, tokenizer, label_to_id, max_len=128):
        self.tokenizer = tokenizer  # 分词器
        self.sentences = sentences  # 文本句子
        self.spo_lists = spo_lists  # 三元组列表
        self.label_to_id = label_to_id  # 标签到ID的映射
        self.max_len = max_len  # 句子的最大长度

    def __len__(self):
        return len(self.sentences)  # 返回数据集的大小

    def __getitem__(self, idx):
        sentence = self.sentences[idx]  # 获取句子
        spo_list = self.spo_lists[idx]  # 获取对应的三元组

        encoded_dict = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_len,
                                      return_tensors='pt')  # 对句子进行编码
        input_id = encoded_dict['input_ids'].squeeze(0)  # 获取输入ID
        attention_mask = encoded_dict['attention_mask'].squeeze(0)  # 获取注意力掩码

        label_id = generate_bio_labels(sentence, spo_list, self.tokenizer, self.label_to_id, self.max_len)

        return {
            'input_ids': input_id,  # 输入ID
            'attention_mask': attention_mask,  # 注意力掩码
            'labels': torch.tensor(label_id)  # 标签ID
        }


# 创建训练、开发和测试数据集
train_dataset_a = RelationExtractionDataset(train_sentences, train_spo_lists, tokenizer_a, label_to_id)  # 训练数据集A
dev_dataset_a = RelationExtractionDataset(dev_sentences, dev_spo_lists, tokenizer_a, label_to_id)  # 开发数据集A
test_dataset_a = RelationExtractionDataset(test_sentences, test_spo_lists, tokenizer_a, label_to_id)  # 测试数据集A

train_dataset_b = RelationExtractionDataset(train_sentences, train_spo_lists, tokenizer_b, label_to_id)  # 训练数据集B
dev_dataset_b = RelationExtractionDataset(dev_sentences, dev_spo_lists, tokenizer_b, label_to_id)  # 开发数据集B
test_dataset_b = RelationExtractionDataset(test_sentences, test_spo_lists, tokenizer_b, label_to_id)  # 测试数据集B

train_dataset_c = RelationExtractionDataset(train_sentences, train_spo_lists, tokenizer_c, label_to_id)  # 训练数据集C
dev_dataset_c = RelationExtractionDataset(dev_sentences, dev_spo_lists, tokenizer_c, label_to_id)  # 开发数据集C
test_dataset_c = RelationExtractionDataset(test_sentences, test_spo_lists, tokenizer_c, label_to_id)  # 测试数据集C

train_dataset_d = RelationExtractionDataset(train_sentences, train_spo_lists, tokenizer_d, label_to_id)  # 训练数据集D
dev_dataset_d = RelationExtractionDataset(dev_sentences, dev_spo_lists, tokenizer_d, label_to_id)  # 开发数据集D
test_dataset_d = RelationExtractionDataset(test_sentences, test_spo_lists, tokenizer_d, label_to_id)  # 测试数据集D

# 使用DataCollatorForTokenClassification进行数据整理
data_collator_a = DataCollatorForTokenClassification(tokenizer_a)
data_collator_b = DataCollatorForTokenClassification(tokenizer_b)
data_collator_c = DataCollatorForTokenClassification(tokenizer_c)
data_collator_d = DataCollatorForTokenClassification(tokenizer_d)

# 模型训练配置
training_args = TrainingArguments(
    output_dir='./results',  # 模型保存路径
    num_train_epochs=12,  # 训练轮数
    per_device_train_batch_size=16,  # 每个设备的训练批量大小
    per_device_eval_batch_size=16,  # 每个设备的评估批量大小
    warmup_steps=500,  # 学习率预热步数
    weight_decay=1,  # 权重衰减
    logging_dir='./logs',  # 日志保存路径
    logging_steps=10,  # 日志记录步数
    evaluation_strategy="epoch",  # 评估策略：每个epoch后评估
    save_strategy="epoch",  # 保存策略：每个epoch后保存
    save_total_limit=2  # 最多保存的模型数量
)


# 计算指标函数
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)  # 预测结果
    labels = p.label_ids  # 实际标签

    preds_flat = preds.flatten()  # 展平预测结果
    labels_flat = labels.flatten()  # 展平实际标签

    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat,
                                                               average='weighted')  # 计算精确度、召回率和F1分数
    acc = accuracy_score(labels_flat, preds_flat)  # 计算准确率

    return {
        'accuracy': acc,  # 返回准确率
        'f1': f1,  # 返回F1分数
        'precision': precision,  # 返回精确度
        'recall': recall  # 返回召回率
    }


# 定义 calculate_micro_macro_averages 函数
def calculate_micro_macro_averages(predictions, labels):
    preds_flat = np.argmax(predictions, axis=2).flatten()  # 将预测结果展平
    labels_flat = labels.flatten()  # 将标签展平

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels_flat, preds_flat,
                                                                                 average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels_flat, preds_flat,
                                                                                 average='macro')

    return {
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }


# 训练和评估模型A
trainer_a = Trainer(
    model=model_a,
    args=training_args,
    train_dataset=train_dataset_a,
    eval_dataset=dev_dataset_a,
    data_collator=data_collator_a,
    compute_metrics=compute_metrics,
)

train_result_a = trainer_a.train()
trainer_a.save_model()
eval_result_a = trainer_a.predict(test_dataset_a)  # 使用 predict 代替 evaluate

# 训练和评估模型B
trainer_b = Trainer(
    model=model_b,
    args=training_args,
    train_dataset=train_dataset_b,
    eval_dataset=dev_dataset_b,
    data_collator=data_collator_b,
    compute_metrics=compute_metrics,
)

train_result_b = trainer_b.train()
trainer_b.save_model()
eval_result_b = trainer_b.predict(test_dataset_b)  # 使用 predict 代替 evaluate

# 训练和评估模型C
trainer_c = Trainer(
    model=model_c,
    args=training_args,
    train_dataset=train_dataset_c,
    eval_dataset=dev_dataset_c,
    data_collator=data_collator_c,
    compute_metrics=compute_metrics,
)

train_result_c = trainer_c.train()
trainer_c.save_model()
eval_result_c = trainer_c.predict(test_dataset_c)  # 使用 predict 代替 evaluate

# 训练和评估模型D
trainer_d = Trainer(
    model=model_d,
    args=training_args,
    train_dataset=train_dataset_d,
    eval_dataset=dev_dataset_d,
    data_collator=data_collator_d,
    compute_metrics=compute_metrics,
)

train_result_d = trainer_d.train()
trainer_d.save_model()
eval_result_d = trainer_d.predict(test_dataset_d)  # 使用 predict 代替 evaluate

# 获取预测结果和标签
predictions_a = eval_result_a.predictions
labels_a = eval_result_a.label_ids
predictions_b = eval_result_b.predictions
labels_b = eval_result_b.label_ids
predictions_c = eval_result_c.predictions
labels_c = eval_result_c.label_ids
predictions_d = eval_result_d.predictions
labels_d = eval_result_d.label_ids

# 计算Micro/Macro Average
micro_macro_averages_a = calculate_micro_macro_averages(predictions_a, labels_a)
micro_macro_averages_b = calculate_micro_macro_averages(predictions_b, labels_b)
micro_macro_averages_c = calculate_micro_macro_averages(predictions_c, labels_c)
micro_macro_averages_d = calculate_micro_macro_averages(predictions_d, labels_d)

# 获取第一个模型的accuracy、F1、runtime、recall、loss、precision。
loss_values_a = trainer_a.state.log_history
eval_accuracy_a = [x['eval_accuracy'] for x in loss_values_a if 'eval_accuracy' in x]
eval_runtime_a = [x['eval_runtime'] for x in loss_values_a if 'eval_runtime' in x]
eval_recall_a = [x['eval_recall'] for x in loss_values_a if 'eval_recall' in x]
eval_loss_a = [x['eval_loss'] for x in loss_values_a if 'eval_loss' in x]
eval_precision_a = [x['eval_precision'] for x in loss_values_a if 'eval_precision' in x]
eval_f1_a = [x['eval_f1'] for x in loss_values_a if 'eval_f1' in x]

# 获取第二个模型的accuracy、F1、runtime、recall、loss、precision。
loss_values_b = trainer_b.state.log_history
eval_accuracy_b = [x['eval_accuracy'] for x in loss_values_b if 'eval_accuracy' in x]
eval_runtime_b = [x['eval_runtime'] for x in loss_values_b if 'eval_runtime' in x]
eval_recall_b = [x['eval_recall'] for x in loss_values_b if 'eval_recall' in x]
eval_loss_b = [x['eval_loss'] for x in loss_values_b if 'eval_loss' in x]
eval_precision_b = [x['eval_precision'] for x in loss_values_b if 'eval_precision' in x]
eval_f1_b = [x['eval_f1'] for x in loss_values_b if 'eval_f1' in x]

# 获取第三个模型的accuracy、F1、runtime、recall、loss、precision。
loss_values_c = trainer_c.state.log_history
eval_accuracy_c = [x['eval_accuracy'] for x in loss_values_c if 'eval_accuracy' in x]
eval_runtime_c = [x['eval_runtime'] for x in loss_values_c if 'eval_runtime' in x]
eval_recall_c = [x['eval_recall'] for x in loss_values_c if 'eval_recall' in x]
eval_loss_c = [x['eval_loss'] for x in loss_values_c if 'eval_loss' in x]
eval_precision_c = [x['eval_precision'] for x in loss_values_c if 'eval_precision' in x]
eval_f1_c = [x['eval_f1'] for x in loss_values_c if 'eval_f1' in x]

# 获取第四个模型的accuracy、F1、runtime、recall、loss、precision。
loss_values_d = trainer_d.state.log_history
eval_accuracy_d = [x['eval_accuracy'] for x in loss_values_d if 'eval_accuracy' in x]
eval_runtime_d = [x['eval_runtime'] for x in loss_values_d if 'eval_runtime' in x]
eval_recall_d = [x['eval_recall'] for x in loss_values_d if 'eval_recall' in x]
eval_loss_d = [x['eval_loss'] for x in loss_values_d if 'eval_loss' in x]
eval_precision_d = [x['eval_precision'] for x in loss_values_d if 'eval_precision' in x]
eval_f1_d = [x['eval_f1'] for x in loss_values_d if 'eval_f1' in x]

# 绘制recall变化曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_recall_a) + 1), eval_recall_a, marker='o', label='Model AnchiBERT recall')
plt.plot(range(1, len(eval_recall_b) + 1), eval_recall_b, marker='o', label='Model guwenbert-base recall')
plt.plot(range(1, len(eval_recall_c) + 1), eval_recall_c, marker='o', label='Model Sikuroberta recall')
plt.plot(range(1, len(eval_recall_d) + 1), eval_recall_d, marker='o', label='Model Bert-ancient-Chinese recall')
plt.xlabel('Epoch')
plt.ylabel('recall')
plt.title('Model recall Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制Loss变化曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_loss_a) + 1), eval_loss_a, marker='o', label='Model AnchiBERT loss')
plt.plot(range(1, len(eval_loss_b) + 1), eval_loss_b, marker='o', label='Model guwenbert-base loss')
plt.plot(range(1, len(eval_loss_c) + 1), eval_loss_c, marker='o', label='Model Sikuroberta loss')
plt.plot(range(1, len(eval_loss_d) + 1), eval_loss_d, marker='o', label='Model Bert-ancient-Chinese loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制accuracy对比图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_accuracy_a) + 1), eval_accuracy_a, marker='o', label='Model AnchiBERT Accuracy')
plt.plot(range(1, len(eval_accuracy_b) + 1), eval_accuracy_b, marker='o', label='Model guwenbert-base Accuracy')
plt.plot(range(1, len(eval_accuracy_c) + 1), eval_accuracy_c, marker='o', label='Model Sikuroberta Accuracy')
plt.plot(range(1, len(eval_accuracy_d) + 1), eval_accuracy_d, marker='o', label='Model Bert-ancient-Chinese Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制runtime对比图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_runtime_a) + 1), eval_runtime_a, marker='o', label='Model AnchiBERT runtime')
plt.plot(range(1, len(eval_runtime_b) + 1), eval_runtime_b, marker='o', label='Model guwenbert-base runtime')
plt.plot(range(1, len(eval_runtime_c) + 1), eval_runtime_c, marker='o', label='Model Sikuroberta runtime')
plt.plot(range(1, len(eval_runtime_d) + 1), eval_runtime_d, marker='o', label='Model Bert-ancient-Chinese runtime')
plt.xlabel('Epoch')
plt.ylabel('runtime')
plt.title('Model runtime Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制F1对比图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_f1_a) + 1), eval_f1_a, marker='o', label='Model AnchiBERT F1')
plt.plot(range(1, len(eval_f1_b) + 1), eval_f1_b, marker='o', label='Model guwenbert-base F1')
plt.plot(range(1, len(eval_f1_c) + 1), eval_f1_c, marker='o', label='Model Sikuroberta F1')
plt.plot(range(1, len(eval_f1_d) + 1), eval_f1_d, marker='o', label='Model Bert-ancient-Chinese F1)')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.title('Model F1 Comparison')
plt.legend()
plt.grid(True)
plt.show()

#绘制precision对比图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_precision_a) + 1), eval_precision_a, marker='o', label='Model AnchiBERT precision')
plt.plot(range(1, len(eval_precision_b) + 1), eval_precision_b, marker='o', label='Model guwenbert-base precision')
plt.plot(range(1, len(eval_precision_c) + 1), eval_precision_c, marker='o', label='Model Sikuroberta precision')
plt.plot(range(1, len(eval_precision_d) + 1), eval_precision_d, marker='o', label='Model Bert-ancient-Chinese precision')
plt.xlabel('Epoch')
plt.ylabel('precision')
plt.title('Model precision Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 模型评估
eval_result_a = trainer_a.evaluate(test_dataset_a)
eval_result_b = trainer_b.evaluate(test_dataset_b)
eval_result_c = trainer_c.evaluate(test_dataset_c)
eval_result_d = trainer_d.evaluate(test_dataset_d)

# 打印结果
#print("--------------AnchiBERT Evaluation Results--------------:", eval_result_a.predictions)
print("--------------AnchiBERT Micro/Macro Averages------------:", micro_macro_averages_a)
#print("--------------guwenbert-base Evaluation Results---------:", eval_result_b.predictions)
print("--------------guwenbert-base Micro/Macro Averages-------:", micro_macro_averages_b)
#print("--------------Sikuroberta Evaluation Results------------:", eval_result_c.predictions)
print("--------------Sikuroberta Micro/Macro Averages----------:", micro_macro_averages_c)
#print("--------------Bert-ancient-Chinese Evaluation Results---:", eval_result_d.predictions)
print("--------------Bert-ancient-Chinese Micro/Macro Averages-:", micro_macro_averages_d)