#!/opt/conda/envs/py310/bin/bash
export PYTHONPATH=$(pwd)
export PATH=/opt/conda/envs/py310/bin:$PATH

for DATASET_NAME in semeval15 semeval16
do
  python -m acsa_add_one.generate_train_dataset -d $DATASET_NAME
  python -m acsa_add_one.train_acsa_add_one -d $DATASET_NAME

  python -m seq2seq.generate_seq2seq_train_dataset -d $DATASET_NAME
  python -m seq2seq.train_seq2seq_model -d $DATASET_NAME

  python -m our_pipeline.generate_dataset_siamese -d $DATASET_NAME
  python -m our_pipeline.train_aspect_extractor -d $DATASET_NAME
  python -m our_pipeline.train_siamese_topic_matcher -d $DATASET_NAME

  python evaluate_all.py -d $DATASET_NAME
  python evaluate_zero_shot.py -d $DATASET_NAME -t semeval15_laptops
  python evaluate_zero_shot.py -d $DATASET_NAME -t semeval16_laptops
done
