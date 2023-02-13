import argparse
from our_pipeline.evaluate import evaluatePipelineSiamese
from seq2seq.evaluate import evaluateS2S
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='semeval16')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'semeval15':
        test_df = 'data/semeval15_test.csv'
        seq2seq_model = 'data/trained_models/seq2seq_semeval15_model/'
        aspect_exrtactor = 'data/trained_models/aspect_extractor_semeval15'
        topic_matcher_siamese = 'data/trained_models/siamese_semeval15_contrastive_loss'

    elif args.dataset == 'semeval16':
        test_df = 'data/semeval16_test.csv'
        seq2seq_model = 'data/trained_models/seq2seq_semeval16_model/'
        aspect_exrtactor = 'data/trained_models/aspect_extractor_semeval16'
        topic_matcher_siamese = 'data/trained_models/siamese_semeval16_contrastive_loss'

    output_seq2seq = evaluateS2S(test_df=test_df, seq2seq_model=seq2seq_model)
    output_seq2seq['approach'] = 'seq2seq'

    output_pipeline_siamese = evaluatePipelineSiamese(test_df=test_df, absa_model=aspect_exrtactor, topic_matcher=topic_matcher_siamese)
    output_pipeline_siamese['approach'] = 'pipeline_siamese'

    out_dict = {key: [elem[key] for elem in [output_seq2seq, output_pipeline_siamese]] for key in output_seq2seq.keys()}

    df_out = pd.DataFrame(out_dict)
    df_out.to_csv('data/results/output_{}.csv'.format(args.dataset), index=False)
