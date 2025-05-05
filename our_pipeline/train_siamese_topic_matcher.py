from sentence_transformers import SentenceTransformer, InputExample, losses
import pandas as pd
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='semeval15')
args = parser.parse_args()

if __name__ == '__main__':
    pre_trained_model = 'all-mpnet-base-v2'

    df_train = pd.read_csv('data/trainset_siamese_{}.csv'.format(args.dataset))
    input_texts = df_train.input_text.tolist()
    topics = df_train.topic.tolist()
    labels = df_train.label.tolist()

    model = SentenceTransformer(pre_trained_model)
    inputs = [InputExample(texts=[input_texts[idx], topics[idx]], label=int(labels[idx])) for idx in range(len(input_texts))]
    dataloader = DataLoader(inputs, shuffle=False, batch_size=16)
    train_loss = losses.ContrastiveLoss(model)
    model.fit(
        train_objectives=[(dataloader, train_loss)], epochs=2, warmup_steps=0,
        output_path='data/trained_models/siamese_{}_contrastive_loss'.format(args.dataset),
        show_progress_bar=True, checkpoint_save_steps=3000)
