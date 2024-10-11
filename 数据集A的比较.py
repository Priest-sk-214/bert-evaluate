import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 定义路径
data_path = '数据集A/'
train_file = os.path.join(data_path, 'train.txt')
dev_file = os.path.join(data_path, 'dev.txt')
test_file = os.path.join(data_path, 'test.txt')

# 加载数据集
def load_data(file_path):
    sentences = []
    labels = []
    sentence = []
    label = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                word, tag = line.strip().split()
                sentence.append(word)
                label.append(tag)
            else:
                if sentence and label:
                    sentences.append(sentence)
                    labels.append(label)
                sentence = []
                label = []
        if sentence and label:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

train_sentences, train_labels = load_data(train_file)
dev_sentences, dev_labels = load_data(dev_file)
test_sentences, test_labels = load_data(test_file)

# 标签集合
unique_labels = set(label for label_list in train_labels for label in label_list)
label_to_id = {label: i for i, label in enumerate(sorted(unique_labels))}
id_to_label = {i: label for label, i in label_to_id.items()}

# 使用AnchiBERT模型的预训练模型和分词器（第一个模型）
config_1 = BertConfig.from_pretrained('./AnchiBERT', num_labels=len(label_to_id))
tokenizer_1 = BertTokenizer.from_pretrained('./AnchiBERT')
model_1 = BertForTokenClassification.from_pretrained('./AnchiBERT', config=config_1)

# 使用GuwenBERT的预训练模型和分词器（第二个模型）
config_2= BertConfig.from_pretrained('./guwenbert-base', num_labels=len(label_to_id))
tokenizer_2 = BertTokenizer.from_pretrained("./guwenbert-base")
model_2 = BertForTokenClassification.from_pretrained("./guwenbert-base", config=config_2)

# 使用Sikuroberta的预训练模型和分词器（第三个模型）
config_3= BertConfig.from_pretrained('./Sikuroberta', num_labels=len(label_to_id))
tokenizer_3= BertTokenizer.from_pretrained('./Sikuroberta')
model_3= BertForTokenClassification.from_pretrained('./Sikuroberta', config=config_3)

# 使用Bert-ancient-Chinese模型的预训练模型和分词器（第四个模型）
config_4= BertConfig.from_pretrained('./Bert-ancient-Chinese', num_labels=len(label_to_id))
tokenizer_4= BertTokenizer.from_pretrained('./Bert-ancient-Chinese')
model_4= BertForTokenClassification.from_pretrained('./Bert-ancient-Chinese', config=config_4)

# 自定义数据集
class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, label_to_id, max_len=128):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoded_dict = self.tokenizer(sentence, truncation=True, is_split_into_words=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        input_id = encoded_dict['input_ids'].squeeze(0)
        attention_mask = encoded_dict['attention_mask'].squeeze(0)

        # 处理标签，确保长度一致
        label_id = [self.label_to_id[l] for l in label]
        if len(label_id) > self.max_len:
            label_id = label_id[:self.max_len]
        else:
            label_id += [self.label_to_id['O']] * (self.max_len - len(label_id))

        return {
            'input_ids': input_id,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_id)
        }

# 创建数据集
train_dataset_1 = NERDataset(train_sentences, train_labels, tokenizer_1, label_to_id)
dev_dataset_1 = NERDataset(dev_sentences, dev_labels, tokenizer_1, label_to_id)
test_dataset_1 = NERDataset(test_sentences, test_labels, tokenizer_1, label_to_id)

train_dataset_2 = NERDataset(train_sentences, train_labels, tokenizer_2, label_to_id)
dev_dataset_2 = NERDataset(dev_sentences, dev_labels, tokenizer_2, label_to_id)
test_dataset_2 = NERDataset(test_sentences, test_labels, tokenizer_2, label_to_id)

train_dataset_3 = NERDataset(train_sentences, train_labels, tokenizer_3, label_to_id)
dev_dataset_3 = NERDataset(dev_sentences, dev_labels, tokenizer_3, label_to_id)
test_dataset_3 = NERDataset(test_sentences, test_labels, tokenizer_3, label_to_id)

train_dataset_4 = NERDataset(train_sentences, train_labels, tokenizer_4, label_to_id)
dev_dataset_4 = NERDataset(dev_sentences, dev_labels, tokenizer_4, label_to_id)
test_dataset_4 = NERDataset(test_sentences, test_labels, tokenizer_4, label_to_id)

# 使用DataCollatorForTokenClassification进行数据整理
data_collator_1 = DataCollatorForTokenClassification(tokenizer_1)
data_collator_2 = DataCollatorForTokenClassification(tokenizer_2)
data_collator_3 = DataCollatorForTokenClassification(tokenizer_3)
data_collator_4 = DataCollatorForTokenClassification(tokenizer_4)

# 模型训练配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=12,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=1,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=4
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    labels = p.label_ids

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='weighted')
    acc = accuracy_score(labels_flat, preds_flat)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels_flat, preds_flat,average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels_flat, preds_flat,average='macro')

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro
    }



# 使用Trainer进行训练
trainer_1 = Trainer(
    model=model_1,
    args=training_args,
    train_dataset=train_dataset_1 ,
    eval_dataset=dev_dataset_1 ,
    data_collator=data_collator_1 ,
    compute_metrics=compute_metrics,
)

trainer_2 = Trainer(
    model=model_2,
    args=training_args,
    train_dataset=train_dataset_2,
    eval_dataset=dev_dataset_2,
    data_collator=data_collator_2,
    compute_metrics=compute_metrics,
)

trainer_3 = Trainer(
    model=model_3,
    args=training_args,
    train_dataset=train_dataset_3,
    eval_dataset=dev_dataset_3,
    data_collator=data_collator_3,
    compute_metrics=compute_metrics,
)

trainer_4 = Trainer(
    model=model_4,
    args=training_args,
    train_dataset=train_dataset_4,
    eval_dataset=dev_dataset_4,
    data_collator=data_collator_4,
    compute_metrics=compute_metrics,
)

# 训练第一个模型并记录损失值
train_result_1 = trainer_1.train()
trainer_1.save_model()  # 保存模型

# 训练第二个模型并记录损失值
train_result_2 = trainer_2.train()
trainer_2.save_model()  # 保存模型

# 训练第三个模型并记录损失值
train_result_3 = trainer_3.train()
trainer_3.save_model()  # 保存模型

# 训练第四个模型并记录损失值
train_result_4 = trainer_4.train()
trainer_4.save_model()  # 保存模型

# 获取第一个模型的accuracy、F1、runtime、recall、loss、precision。
loss_values_1 = trainer_1.state.log_history
eval_accuracy_1 = [x['eval_accuracy'] for x in loss_values_1 if 'eval_accuracy' in x]
eval_runtime_1 = [x['eval_runtime'] for x in loss_values_1 if 'eval_runtime' in x]
eval_recall_1 = [x['eval_recall'] for x in loss_values_1 if 'eval_recall' in x]
eval_loss_1 = [x['eval_loss'] for x in loss_values_1 if 'eval_loss' in x]
eval_precision_1 = [x['eval_precision'] for x in loss_values_1 if 'eval_precision' in x]
eval_f1_1 = [x['eval_f1'] for x in loss_values_1 if 'eval_f1' in x]

# 获取第二个模型的accuracy、F1、runtime、recall、loss、precision。
loss_values_2 = trainer_2.state.log_history
eval_accuracy_2 = [x['eval_accuracy'] for x in loss_values_2 if 'eval_accuracy' in x]
eval_runtime_2 = [x['eval_runtime'] for x in loss_values_2 if 'eval_runtime' in x]
eval_recall_2 = [x['eval_recall'] for x in loss_values_2 if 'eval_recall' in x]
eval_loss_2 = [x['eval_loss'] for x in loss_values_2 if 'eval_loss' in x]
eval_precision_2 = [x['eval_precision'] for x in loss_values_2 if 'eval_precision' in x]
eval_f1_2 = [x['eval_f1'] for x in loss_values_2 if 'eval_f1' in x]

# 获取第三个模型的accuracy、F1、runtime、recall、loss、precision。
loss_values_3 = trainer_3.state.log_history
eval_accuracy_3 = [x['eval_accuracy'] for x in loss_values_3 if 'eval_accuracy' in x]
eval_runtime_3 = [x['eval_runtime'] for x in loss_values_3 if 'eval_runtime' in x]
eval_recall_3 = [x['eval_recall'] for x in loss_values_3 if 'eval_recall' in x]
eval_loss_3 = [x['eval_loss'] for x in loss_values_3 if 'eval_loss' in x]
eval_precision_3 = [x['eval_precision'] for x in loss_values_3 if 'eval_precision' in x]
eval_f1_3 = [x['eval_f1'] for x in loss_values_3 if 'eval_f1' in x]

# 获取第四个模型的accuracy、F1、runtime、recall、loss、precision。
loss_values_4 = trainer_4.state.log_history
eval_accuracy_4 = [x['eval_accuracy'] for x in loss_values_4 if 'eval_accuracy' in x]
eval_runtime_4 = [x['eval_runtime'] for x in loss_values_4 if 'eval_runtime' in x]
eval_recall_4 = [x['eval_recall'] for x in loss_values_4 if 'eval_recall' in x]
eval_loss_4 = [x['eval_loss'] for x in loss_values_4 if 'eval_loss' in x]
eval_precision_4 = [x['eval_precision'] for x in loss_values_4 if 'eval_precision' in x]
eval_f1_4 = [x['eval_f1'] for x in loss_values_4 if 'eval_f1' in x]

# 绘制recall变化曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_recall_1) + 1), eval_recall_1, marker='o', label='Model AnchiBERT recall')
plt.plot(range(1, len(eval_recall_2) + 1), eval_recall_2, marker='o', label='Model guwenbert-base recall')
plt.plot(range(1, len(eval_recall_3) + 1), eval_recall_3, marker='o', label='Model Sikuroberta recall')
plt.plot(range(1, len(eval_recall_4) + 1), eval_recall_4, marker='o', label='Model Bert-ancient-Chinese recall')
plt.xlabel('Epoch')
plt.ylabel('recall')
plt.title('Model recall Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制Loss变化曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_loss_1) + 1), eval_loss_1, marker='o', label='Model AnchiBERT loss')
plt.plot(range(1, len(eval_loss_2) + 1), eval_loss_2, marker='o', label='Model guwenbert-base loss')
plt.plot(range(1, len(eval_loss_3) + 1), eval_loss_3, marker='o', label='Model Sikuroberta loss')
plt.plot(range(1, len(eval_loss_4) + 1), eval_loss_4, marker='o', label='Model Bert-ancient-Chinese loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制accuracy对比图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_accuracy_1) + 1), eval_accuracy_1, marker='o', label='Model AnchiBERT Accuracy')
plt.plot(range(1, len(eval_accuracy_2) + 1), eval_accuracy_2, marker='o', label='Model guwenbert-base Accuracy')
plt.plot(range(1, len(eval_accuracy_3) + 1), eval_accuracy_3, marker='o', label='Model Sikuroberta Accuracy')
plt.plot(range(1, len(eval_accuracy_4) + 1), eval_accuracy_4, marker='o', label='Model Bert-ancient-Chinese Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制runtime对比图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_runtime_1) + 1), eval_runtime_1, marker='o', label='Model AnchiBERT runtime')
plt.plot(range(1, len(eval_runtime_2) + 1), eval_runtime_2, marker='o', label='Model guwenbert-base runtime')
plt.plot(range(1, len(eval_runtime_3) + 1), eval_runtime_3, marker='o', label='Model Sikuroberta runtime')
plt.plot(range(1, len(eval_runtime_4) + 1), eval_runtime_4, marker='o', label='Model Bert-ancient-Chinese runtime')
plt.xlabel('Epoch')
plt.ylabel('runtime')
plt.title('Model runtime Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 绘制F1对比图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_f1_1) + 1), eval_f1_1, marker='o', label='Model AnchiBERT F1')
plt.plot(range(1, len(eval_f1_2) + 1), eval_f1_2, marker='o', label='Model guwenbert-base F1')
plt.plot(range(1, len(eval_f1_3) + 1), eval_f1_3, marker='o', label='Model Sikuroberta F1')
plt.plot(range(1, len(eval_f1_4) + 1), eval_f1_4, marker='o', label='Model Bert-ancient-Chinese F1)')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.title('Model F1 Comparison')
plt.legend()
plt.grid(True)
plt.show()

#绘制precision对比图
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(eval_precision_1) + 1), eval_precision_1, marker='o', label='Model AnchiBERT precision')
plt.plot(range(1, len(eval_precision_2) + 1), eval_precision_2, marker='o', label='Model guwenbert-base precision')
plt.plot(range(1, len(eval_precision_3) + 1), eval_precision_3, marker='o', label='Model Sikuroberta precision')
plt.plot(range(1, len(eval_precision_4) + 1), eval_precision_4, marker='o', label='Model Bert-ancient-Chinese precision')
plt.xlabel('Epoch')
plt.ylabel('precision')
plt.title('Model precision Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 模型评估
eval_result_1 = trainer_1.evaluate(test_dataset_1)
eval_result_2 = trainer_2.evaluate(test_dataset_2)
eval_result_3 = trainer_3.evaluate(test_dataset_3)
eval_result_4 = trainer_4.evaluate(test_dataset_4)

# 显示评估结果
print("-------------------AnchiBERT--------------------------A------------")
print("Model AnchiBERT Evaluation Results:", eval_result_1)

print("-------------------guwenbert-base---------------------A------------")
print("Model guwenbert-base Evaluation Results:", eval_result_2)

print("-------------------Sikuroberta------------------------A------------")
print("Model Sikuroberta Evaluation Results:", eval_result_3)

print("-------------------Bert-ancient-Chinese---------------A------------")
print("Model Bert-ancient-Chinese Evaluation Results:", eval_result_4)
