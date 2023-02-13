from our_pipeline.pipeline_predict_class_siamese import AbsaPredictPipelineSiamese
from utils import (computeMetrics, processTestDf)


def evaluatePipelineSiamese(
        test_df='data/homesupply_test.csv',
        absa_model='data/trained_models/aspect_extractor_homesupply',
        topic_matcher='data/trained_models/siamese_homesupply_contrastive_loss',
        level='aspect_category'
):
    labels_mapping = {'T-POS': 'POSITIVE', 'T-NEG': 'NEGATIVE',
                      'T-NEU': 'NEUTRAL'}

    def processResultLine(tuples_list):
        return [(elem[0], labels_mapping[elem[2]]) if elem else [] for elem in
                tuples_list]

    texts, gold_labels, possible_labels_list = processTestDf(
        test_df, level=level)

    model = AbsaPredictPipelineSiamese(
        absa_model=absa_model, topic_match_model=topic_matcher,
        topics_list=possible_labels_list)
    predictions = model.predict(texts)
    processed_preds = [processResultLine(elem) for elem in predictions]
    pred_lengths = [len(elem) for elem in processed_preds]
    out = computeMetrics(gold_labels, processed_preds, possible_labels_list)
    avg_preds = sum(pred_lengths) / len(pred_lengths)
    out['avg_preds'] = avg_preds
    print(out)
    return out

if __name__ == '__main__':
    evaluatePipelineSiamese()
