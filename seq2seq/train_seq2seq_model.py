from t5_generator import T5Generator
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='semeval15')
args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv('data/trainset_seq2seq_{}.csv'.format(args.dataset))
    pre_trained_model = 'google/t5-v1_1-small'
    model = T5Generator(pre_trained_model)
    input_text = df.input_string.tolist()
    output_text = df.output_string.tolist()
    model.fit(
        input_text, output_text,
        output_path='data/trained_models/seq2seq_{}_model'.format(args.dataset),
        per_device_train_batch_size=9, learning_rate=0.0001, epochs=10, seed=42,
        early_stop=False)
