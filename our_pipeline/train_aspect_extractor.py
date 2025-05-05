from aspect_extractor_class import Absa
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='bert-base-uncased')
parser.add_argument('-d', '--dataset', type=str, default='semeval15')
parser.add_argument('--epochs', type=int, default=2)
args = parser.parse_args()

output_path = 'data/trained_models/aspect_extractor_{}'.format(args.dataset)
absa_model = Absa(project='absa_paper')
absa_model.dataset_path = 'data/'
absa_model.base_model = args.model
absa_model.params.epochs = args.epochs
absa_model.max_length = 256
absa_model.params.batch_size = 16
absa_model.train('{}_train.csv'.format(args.dataset), output_dir=output_path)
