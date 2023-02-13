import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='semeval15')
args = parser.parse_args()

if __name__ == '__main__':

    random.seed(101)

    def selectNegativeExamples(true_label, labels_list, n_negatives=7):
        available_labels = [elem for elem in labels_list if elem != true_label]
        negative_examples = []
        for _ in range(n_negatives):
            negative_examples.append(available_labels.pop(random.randrange(
                len(available_labels))))
            if not available_labels:
                break
        return negative_examples

    def composeTrainInputs(sentence, negative_examples, term, true_topic):
        template = "{sentence} [SEP] {term}"
        examples = [template.format(sentence=sentence, term=term)]
        topics = [true_topic]
        labels = [1.0]
        for negative_ex in negative_examples:
            examples.append(template.format(sentence=sentence, term=term))
            topics.append(negative_ex)
            labels.append(0.0)
        return examples, topics, labels

    train_data = 'data/{}_train.csv'.format(args.dataset)
    df_train_data = pd.read_csv(train_data)
    possible_labels_list = df_train_data['aspect_category'].unique().tolist()
    examples = []
    topics = []
    labels = []
    for idx, row in df_train_data.iterrows():
        current_negatives = selectNegativeExamples(
            row['aspect_category'], possible_labels_list)
        current_examples, current_topics, current_labels = composeTrainInputs(
            row['text'], current_negatives, row['target'], row['aspect_category'])
        examples += current_examples
        topics += current_topics
        labels += current_labels
    df_out = pd.DataFrame(
        {'input_text': examples, 'topic': topics, 'label': labels})
    df_out = df_out.sample(frac=1, random_state=101)
    df_out.to_csv('data/trainset_siamese_{}.csv'.format(args.dataset), index=False)
