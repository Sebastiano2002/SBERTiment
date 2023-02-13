import argparse
from our_pipeline.evaluate import evaluatePipelineSiamese
from acsa_add_one.evaluate import evaluateACSA
from seq2seq.evaluate import evaluateS2S
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='semeval16')
parser.add_argument('-t', '--testset', type=str,  default='semeval16_laptops')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'semeval16':
        acsa_model = 'data/trained_models/acsa_add_one_semeval16_model/'
        seq2seq_model = 'data/trained_models/seq2seq_semeval16_model/'

        aspect_exrtactor = 'data/trained_models/aspect_extractor_semeval16/'
        topic_matcher_siamese = 'data/trained_models/siamese_semeval16_contrastive_loss/'

    elif args.dataset == 'semeval15':
        acsa_model = 'data/trained_models/acsa_add_one_semeval15_model/'
        seq2seq_model = 'data/trained_models/seq2seq_semeval15_model/'

        aspect_exrtactor = 'data/trained_models/aspect_extractor_semeval15'
        topic_matcher_siamese = 'data/trained_models/siamese_semeval15_contrastive_loss'
    else:
        raise ValueError('Unacceptable dataset')

    if args.testset == 'semeval15_laptops':
        test_df = 'data/semeval15_laptops_test.csv'
    elif args.testset == 'semeval16_laptops':
        test_df = 'data/semeval16_laptops_test.csv'

    output_acsa = evaluateACSA(test_df=test_df, acsa_model=acsa_model)
    output_acsa['approach'] = 'acsa'

    output_seq2seq = evaluateS2S(test_df=test_df, seq2seq_model=seq2seq_model)
    output_seq2seq['approach'] = 'seq2seq'

    output_pipeline_siamese = evaluatePipelineSiamese(test_df=test_df, absa_model=aspect_exrtactor, topic_matcher=topic_matcher_siamese)
    output_pipeline_siamese['approach'] = 'pipeline_siamese'


    out_dict = {key: [elem[key] for elem in [output_acsa, output_seq2seq, output_pipeline_siamese]] for key in output_acsa.keys()}
    df_out = pd.DataFrame(out_dict)
    df_out.to_csv(f'data/results/output_{args.dataset}_{args.testset}_zeroshot.csv', index=False)
