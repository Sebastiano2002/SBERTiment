from seq2seq.seq2seq_predict_class import Seq2seqModelPredict
from utils import computeMetrics, processTestDf

def evaluateS2S(
        test_df='data/homesupply_test.csv',
        seq2seq_model='data/trained_models/seq2seq_homesupply_model/',
        level='aspect_category'
):
    texts, gold_labels, possible_labels_list = processTestDf(
        test_df, level=level)
    model = Seq2seqModelPredict(seq2seq_model)
    predictions = model.predict(texts)
    pred_lengths = [len(elem) for elem in predictions]
    out = computeMetrics(gold_labels, predictions, possible_labels_list)
    avg_preds = sum(pred_lengths) / len(pred_lengths)
    out['avg_preds'] = avg_preds
    print('S2S results')
    print(out)
    return out

if __name__ == '__main__':
    evaluateS2S()
