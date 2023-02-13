from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter, OrderedDict
import pandas as pd


def extractData(df, level='aspect_category'):
    final_res_list = []
    unique_sampleids = list(set(df.sample_id.tolist()))
    for id_row in unique_sampleids:
        subdf = df[df['sample_id'] == id_row]
        text = subdf.text.tolist()[0]
        subtopics_sentiments = []
        for idx, row in subdf.iterrows():
            subtopics_sentiments.append((row[level], row['sentiment']))
        final_res_list.append((id_row, text, subtopics_sentiments))
    return final_res_list


def extractTextGoldLabels(df, level='aspect_category'):
    out = extractData(df, level=level)
    temp = list(zip(*out))
    texts = [str(elem) for elem in temp[1]]
    gold_labels = temp[2]
    return texts, gold_labels


def extractTopicsFromInputs(true_labels_list, preds_list):
    true_subtopics_list = [[elem[0] for elem in row_labels] for row_labels in
                           true_labels_list]
    preds_subtopics_list = [[elem[0] for elem in row_labels] for row_labels in
                            preds_list]
    return true_subtopics_list, preds_subtopics_list


def processTestDf(test_df, level=''):
    df_test = pd.read_csv(test_df)
    texts, gold_labels = extractTextGoldLabels(df_test, level='aspect_category')
    possible_labels_list = df_test['aspect_category'].unique().tolist()
    return texts, gold_labels, possible_labels_list


def computeMicroScores(
        true_labels_list, preds_list, possible_labels_list):
    true_topics_list, preds_topics_list = extractTopicsFromInputs(true_labels_list, preds_list)
    dv = DictVectorizer()
    dv.fit([OrderedDict.fromkeys(possible_labels_list, 1)])
    def convertToBOWMatrix(labels_list):
        bow_matrix = dv.transform(Counter(element) for element in labels_list).A.astype(int)
        bow_matrix[bow_matrix > 0] = 1
        return bow_matrix

    bow_gold = convertToBOWMatrix(true_topics_list)
    bow_preds = convertToBOWMatrix(preds_topics_list)
    return f1_score(bow_gold, bow_preds, average='micro'), precision_score(bow_gold, bow_preds, average='micro'), recall_score(bow_gold, bow_preds, average='micro')


def computeSentimentsMicroScores(
        true_labels_list, preds_list, possible_labels_list):
    def processLabelsOutput(labels_list):
        output_list = [[elem[0].lower() + '#' + elem[1].lower() for elem in row_labels] for row_labels in labels_list]
        return output_list
    possible_labels_list_w_sentiments = []
    for elem in possible_labels_list:
        possible_labels_list_w_sentiments.append(elem.lower() + '#positive')
        possible_labels_list_w_sentiments.append(elem.lower() + '#negative')
        possible_labels_list_w_sentiments.append(elem.lower() + '#neutral')
    processed_true_labels = processLabelsOutput(true_labels_list)
    processed_preds_list = processLabelsOutput(preds_list)
    dv = DictVectorizer()
    dv.fit([OrderedDict.fromkeys(possible_labels_list_w_sentiments, 1)])
    def convertToBOWMatrix(labels_list):
        bow_matrix = dv.transform(Counter(element) for element in labels_list).A.astype(int)
        bow_matrix[bow_matrix > 0] = 1
        return bow_matrix

    bow_gold = convertToBOWMatrix(processed_true_labels)
    bow_preds = convertToBOWMatrix(processed_preds_list)
    return f1_score(bow_gold, bow_preds, average='micro'), precision_score(bow_gold, bow_preds, average='micro'), recall_score(bow_gold, bow_preds, average='micro')


def computeMetrics(true_labels_list, preds_list, possible_labels_list):
    f1_micro_cat, precision_micro_cat, recall_micro_cat = computeMicroScores(true_labels_list, preds_list, possible_labels_list)
    f1_micro_sen, precision_micro_sen, recall_micro_sen = computeSentimentsMicroScores(true_labels_list, preds_list, possible_labels_list)

    return {
        'f1_micro_category': f1_micro_cat,
        'precision_micro_category': precision_micro_cat,
        'recall_micro_category': recall_micro_cat,
        'f1_micro_sentiment': f1_micro_sen,
        'precision_micro_sentiment': precision_micro_sen,
        'recall_micro_sentiment': recall_micro_sen,
    }
