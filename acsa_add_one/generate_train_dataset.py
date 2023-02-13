import pandas as pd
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_max', type=int, default=7)
parser.add_argument('-d', '--dataset', type=str, default='semeval15')
args = parser.parse_args()

if __name__ == '__main__':
    n_max = args.n_max
    random.seed(101)
    df = pd.read_csv('data/{}_train.csv'.format(args.dataset))
    subtopic_list = list(set(df.aspect_category))
    input_template = "{category} [SEP] {text}"

    def selectNegativeExamples(categories_list, true_categories, n_negative_examples=5):
        selected_negatives = []
        available_categories = [elem for elem in categories_list if elem not in true_categories]
        for _ in range(n_negative_examples):
            selected_negative = available_categories.pop(random.randrange(len(available_categories)))
            selected_negatives.append(selected_negative)
            if not available_categories:
                break
        return selected_negatives

    input_sentences = []
    labels = []
    sample_ids = list(set(df.sample_id))
    for sampleid in sample_ids:
        subdf = df[df['sample_id'] == sampleid]
        curr_sentence = subdf['text'].tolist()[0]
        true_categories = subdf['aspect_category'].tolist()
        curr_sentiments = subdf['sentiment'].tolist()
        n_negatives_to_sample = n_max - len(subdf) if n_max - len(subdf) > 0 else 1
        negatives = selectNegativeExamples(subtopic_list, true_categories=true_categories, n_negative_examples=n_negatives_to_sample)
        for idx in range(len(true_categories)):
            input_sentences.append(input_template.format(text=curr_sentence, category=true_categories[idx]))
            labels.append(curr_sentiments[idx])
        for negative_example in negatives:
            input_sentences.append(input_template.format(category=negative_example, text=curr_sentence))
            labels.append('no')

    out_df = pd.DataFrame({'input_sentences': input_sentences, 'labels': labels})
    out_df = out_df.sample(frac=1, random_state=101)
    out_df.to_csv('data/trainset_acsa_add_one_{}.csv'.format(args.dataset), index=False)
