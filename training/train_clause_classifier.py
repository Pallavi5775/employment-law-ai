# training/train_clause_classifier.py
# Minimal runnable training stub using transformers for clause classification.
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
import torch
import argparse

LABELS = ['CONFIDENTIALITY','NON_COMPETE','TERMINATION','SEVERANCE','NOTICE_PERIOD','DISCRIMINATION','IP_ASSIGNMENT','ARBITRATION','GOVERNING_LAW','MISSING_CLAUSE']

def load_dataset(path=None):
    # placeholder: implement dataset loading from Delta/CSV
    data = {'text': ['The employee shall not compete...','This agreement may be terminated...'], 'label':[1,2]}
    return datasets.Dataset.from_dict(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='distilbert-base-uncased')
    parser.add_argument('--output', default='models/clause_model')
    args = parser.parse_args()

    ds = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tok(batch): return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)
    ds = ds.map(tok, batched=True)
    ds = ds.train_test_split(test_size=0.2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABELS))

    training_args = TrainingArguments(output_dir=args.output, per_device_train_batch_size=8, num_train_epochs=1)
    trainer = Trainer(model=model, args=training_args, train_dataset=ds['train'], eval_dataset=ds['test'])
    trainer.train()
    trainer.save_model(args.output)

if __name__ == '__main__':
    main()
