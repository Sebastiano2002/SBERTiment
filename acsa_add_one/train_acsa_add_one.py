import pandas as pd
from bert_classifier import BertClassifier
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='semeval15')
args = parser.parse_args()

if __name__ == '__main__':
    pre_trained_model = "bert-base-uncased"
    df = pd.read_csv('data/trainset_acsa_add_one_{}.csv'.format(args.dataset))
    texts = df['input_sentences'].tolist()
    labels = df['labels'].tolist()
    model = BertClassifier(pre_trained_model)
    model.fit(
        texts, labels, output_path='data/trained_models/acsa_add_one_{}_model'.format(args.dataset),
        per_device_train_batch_size=16, learning_rate=0.0001, epochs=3,
        early_stop=False, newtrain=True)
