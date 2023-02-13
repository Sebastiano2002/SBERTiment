import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

from our_pipeline.absa_utils import (AbsaBertParams,
                                                CustomStoppingCallback,
                                                InputExample, Prediction,
                                                TokenClassificationDataset,
                                                TokenClassificationTask,
                                                computeAbsaMetrics,
                                                dfToExamples,
                                                fromWordsToAspects,
                                                platformToExamples, splitWords,
                                                txtToExamples)

logger = logging.getLogger(__name__)


class Absa:
    labels = ["T-POS", "T-NEG", "T-NEU", "O"]

    def __init__(self, project, lang='multi', cache_model=False):
        self.max_seq_length = 128

        config_branch = {
            'dataset_path': 'absa_paper/data/',
            'output_dir': 'absa_paper/absa_model',
            'default_model': 'bert-base-uncased'
        }

        self.params = AbsaBertParams()
        self.dataset_path = config_branch['dataset_path']

        self.model_name = os.path.join(config_branch['output_dir'],
                                       f"{project}_{lang}")
        self.cached_model = None
        self.cached_tokenizer = None

        if os.path.isdir(self.model_name) \
                and 'config.json' in os.listdir(self.model_name):
            self.base_model = self.model_name

            if cache_model:
                logger.info(f"Caching trained model at {self.model_name}")
                self.cached_model = self._loadModel(self.model_name)
                self.cached_tokenizer = \
                    AutoTokenizer.from_pretrained(self.base_model)
        else:
            self.base_model = config_branch['default_model']

            if cache_model:
                self.cached_model = self._loadModel(self.base_model)
                self.cached_tokenizer = \
                    AutoTokenizer.from_pretrained(self.base_model)

    def _loadExamples(self, dataset_file, tokenizer, type_df='platform',
                      split=False):
        """Load dataset from a file
        csv or tsv with columns:
            survey: text
            aspect1: word or group of words
            sentiment1: POS, NEG or NEU
            ...
            aspectN, sentimentN

        txt formatted with the OT schema
            each row represents an entry
            first the whole sentence
            then after #### word=label separed by spaces
            eg: Ottima batteria####Ottima=O batteria=T-POS

        Returns a tuple 2 lists of InputExample
        """
        file_path = os.path.join(self.dataset_path, dataset_file)
        if file_path.endswith('.txt'):
            examples = txtToExamples(file_path)
        else:
            if file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise IOError('Invalid file format')

            if type_df == 'platform':
                examples = platformToExamples(df, tokenizer)

            else:
                n_aspects = len([d for d in df.columns
                                 if d.startswith('aspect')])
                examples = dfToExamples(df, tokenizer, n_aspects=n_aspects)

        if split:
            train_examples, test_examples = train_test_split(
                examples, test_size=0.2, random_state=42
            )
        else:
            train_examples = examples
            test_examples = examples

        return train_examples, test_examples

    def _examplesToDataset(self, examples, tokenizer):
        """Convert a list of InputExample in a Dataset object"""
        dataset = TokenClassificationDataset(
            token_classification_task=TokenClassificationTask(),
            examples=examples,
            tokenizer=tokenizer,
            labels=self.labels,
            max_seq_length=self.max_seq_length,
            overwrite_cache=True,
        )
        return dataset

    def _setupTrainer(self, model, early_stop=False,
                      train_dataset=None, output_dir=None):
        """Given the model instantiate an huggingface Trainer object"""
        # Trainer requires a valid output dir even if it is not training
        if not output_dir:
            output_dir = self.model_name

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            do_train=True,
            per_device_train_batch_size=self.params.batch_size,
            learning_rate=self.params.learning_rate,
            num_train_epochs=self.params.epochs,
            save_strategy='no',
            seed=self.params.seed,
            no_cuda=self.device == torch.device('cpu'),

            logging_strategy='epoch',
        )

        self.stopping_callback = CustomStoppingCallback(patience=2)

        return Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            callbacks=[self.stopping_callback] if early_stop else [],
        )

    def _loadModel(self, model_name):
        """Load a model from its name or path"""
        bert_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label={i: label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)},
        )

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=bert_config
        )

        return model

    def train(self, dataset, split=False, early_stop=True, epochs=None,
              base_model=None, output_dir=None, type_df='platform'):
        """Train a model, returns the number of epochs required"""

        if base_model:
            self.base_model = base_model

        if not output_dir:
            output_dir = self.model_name

        if self.cached_model:
            model = self.cached_model
            tokenizer = self.cached_tokenizer
        else:
            model = self._loadModel(self.base_model)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        if epochs:
            self.params.epochs = epochs

        self._setDevice()
        train_examples, test_examples = self._loadExamples(
            dataset, tokenizer, type_df, split
        )

        train_dataset = self._examplesToDataset(
            train_examples, tokenizer
        )

        trainer = self._setupTrainer(
            model, early_stop, train_dataset, output_dir
        )

        model.to(self.device)
        trainer.train()

        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir=output_dir)
        tokenizer.save_pretrained(output_dir)

        if self.cached_model:
            self.cache_model = model
            self.cached_tokenizer = tokenizer

        if early_stop:
            epochs = self.stopping_callback.epochs
            logger.info(f"Stopped at {epochs} epochs")
        else:
            epochs = self.params.epochs
        return epochs

    def _alignPredictions(self,
                          predictions: np.ndarray,
                          label_ids: np.ndarray
                          ) -> Tuple[List[str], List[str]]:
        """Convert tensors in list of labels and confidences"""
        label_map: Dict[int, str] = {i: label
                                     for i, label in enumerate(self.labels)}

        # TODO: check this command
        # trainer.predict returns a tuple, the first element should be a
        # tensor, but on some setups it is another tuple, the first element of
        # which is the tensor
        if isinstance(predictions, tuple):
            predictions, _ = predictions

        confs = np.max(predictions, axis=2)
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        confs_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
                    confs_list[i].append(confs[i][j])

        return preds_list, out_label_list, confs_list

    def predict(self, sentences, model_name=None, cache_model=False):
        """Take a list of sentences returns a list of Prediction protos"""

        if not model_name and cache_model and self.cached_model:
            model = self.cached_model
            tokenizer = self.cached_tokenizer
        else:
            model = self._loadModel(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if not model_name:
                model_name = self.model_name

        if cache_model:
            self.cached_model = model

        self._setDevice()
        trainer = self._setupTrainer(model)

        examples = [InputExample(guid=i, text=text,
                                 words=splitWords(text, tokenizer),
                                 labels=['O'] * self.max_seq_length)
                    for i, text in enumerate(sentences)]

        dataset = self._examplesToDataset(
            examples, tokenizer
        )

        model.to(self.device)

        predictions_tensors, label_ids, metrics = \
            trainer.predict(dataset)

        preds_list, _, confs_list = \
            self._alignPredictions(predictions_tensors, label_ids)
        # pred_list is already a list of labels for each word

        predictions = []
        for ex, word_labels, word_confs in \
                zip(examples, preds_list, confs_list):
            words = splitWords(ex.text, tokenizer)

            # handle truncation
            if len(word_labels) < len(words):
                delta_len = len(words) - len(word_labels)
                word_labels.extend(['O'] * delta_len)
                word_confs.extend([0.0] * delta_len)

            aspects = fromWordsToAspects(words, word_labels, word_confs)
            predictions.append(
                Prediction(words=words, labels=word_labels, aspects=aspects)
            )

        return predictions

    def evaluate(self, dataset, model_name=None, split=False,
                 misclassified_path=None, type_df='platform'):
        """Evaluate a model, return metric computed by
        absa.utils.computeAbsaMetrics()"""

        if not model_name and self.cached_model:
            model = self.cached_model
            tokenizer = self.cached_tokenizer
        else:
            model = self._loadModel(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if not model_name:
                model_name = self.model_name

        self._setDevice()
        train_examples, test_examples = \
            self._loadExamples(dataset, tokenizer, type_df, split)

        trainer = self._setupTrainer(model)

        test_dataset = self._examplesToDataset(
            test_examples, tokenizer
        )

        predictions_tensors, label_ids, metrics = \
            trainer.predict(test_dataset)

        preds_list, _, _ = \
            self._alignPredictions(predictions_tensors, label_ids)

        # handle truncation padding with 'O' tag
        for word_labels, ex in zip(preds_list, test_examples):
            if len(word_labels) < len(ex.labels):
                delta_len = len(ex.labels) - len(word_labels)
                word_labels.extend(['O'] * delta_len)

        @dataclass
        class Prediction:
            words: List[str]
            labels: Optional[List[str]]

        predictions = [Prediction(words=ex.words, labels=labels)
                       for (ex, labels) in zip(test_examples, preds_list)]

        metrics = computeAbsaMetrics(
            test_examples, predictions, misclassified_path
        )

        metrics.update({
            'train_len': len(train_examples),
            'test_len': len(test_examples),
        })
        return metrics

    def _setDevice(self, device=None):
        """Set torch device to cuda or cpu according to the configuration"""
        if device is not None:
            self.device = torch.device(device)
            return

        if self.params.device == "auto":
            self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self.device = torch.device(self.params.device)
