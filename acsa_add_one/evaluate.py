from acsa_add_one.acsa_predict_class import ACSAPredictor
from utils import computeMetrics
from utils import processTestDf


def evaluateACSA(
        test_df='data/semeval_16_test.csv',
        acsa_model='data/trained_models/acsa_add_one_semeval16/',
        level='aspect_category'
):
    texts, gold_labels, possible_labels_list = processTestDf(
        test_df, level=level)
    model = ACSAPredictor(acsa_model, possible_labels_list)
    predictions = model.predictSentences(sentences=texts)
    pred_lengths = [len(elem) for elem in predictions]
    out = computeMetrics(gold_labels, predictions, possible_labels_list)
    avg_preds = sum(pred_lengths)/len(pred_lengths)
    out['avg_preds'] = avg_preds
    print('ACSA results')
    print(out)
    return out


if __name__ == '__main__':
    evaluateACSA()
