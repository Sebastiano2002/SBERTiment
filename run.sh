#!/opt/conda/envs/py310/bin/python
for DATASET_NAME in semeval15 semeval16
do
  python3 -m acsa_add_one.generate_train_dataset -d $DATASET_NAME
  python3 -m acsa_add_one.train_acsa_add_one -d $DATASET_NAME

  python3 -m seq2seq.generate_seq2seq_train_dataset -d $DATASET_NAME
  python3 -m seq2seq.train_seq2seq_model -d $DATASET_NAME

  python3 -m our_pipeline.generate_dataset_siamese -d $DATASET_NAME
  python3 -m our_pipeline.train_aspect_extractor -d $DATASET_NAME
  python3 -m our_pipeline.train_siamese_topic_matcher -d $DATASET_NAME

  python3 -m evaluate_all -d $DATASET_NAME
  python3 -m evaluate_zero_shot -d $DATASET_NAME -t semeval15_laptops
  python3 -m evaluate_zero_shot -d $DATASET_NAME -t semeval16_laptops
done
