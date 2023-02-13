import logging

import torch
from torch.utils.data.dataset import Dataset
from transformers import (Trainer, TrainingArguments,
                          T5ForConditionalGeneration, T5Tokenizer)

from bert_classifier import CustomStoppingCallback


class ProjectDataset(Dataset):
    def __init__(self, encodings, label_encodings):
        self.encodings = encodings
        self.label_encodings = label_encodings

    def __len__(self):
        return len(self.label_encodings)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = self.label_encodings[idx]
        return item


class T5Generator:
    NAME = "t5"
    default_max_length = 256

    def __init__(self, pre_trained_model):
        self.pre_trained_model = pre_trained_model
        self.generator = None
        self.tokenizer = None
        self.trainer = None
        self.device = 'auto'
        self.fp16 = False

    def toFp16(self):
        self.fp16 = True
        if self.generator is not None:
            self.generator.half()

    def _encodeSentences(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        max_length = self.default_max_length
        encoded_sentences = self.tokenizer(
            sentences, add_special_tokens=True, padding="max_length",
            truncation=True, max_length=max_length, return_tensors='pt')

        if encoded_sentences.get("token_type_ids"):
            encoded_sentences.pop("token_type_ids")
        return encoded_sentences

    def _setDevice(self, device=None):
        if device is not None and device != 'auto':
            self.device = torch.device(device)
            return

        if device == "auto":
            self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self.device = torch.device(self.device)

    def _definePath(self, path):
        if not path.endswith("/"):
            path += "/"
        return path

    def fit(self, sentences, label_strings, output_path=None,
            device='auto', per_device_train_batch_size=32, learning_rate=10e-5,
            epochs=10, seed=42, save=True, early_stop=False):
        path = self._definePath(output_path)

        self._setDevice(device)
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.pre_trained_model, use_fast=True)
        self.generator = T5ForConditionalGeneration.from_pretrained(
            self.pre_trained_model)
        self.generator.to(self.device)

        self.generator.train()

        encoded_sentences = self._encodeSentences(sentences)
        encoded_labels = self._encodeSentences(label_strings).input_ids
        encoded_labels = torch.tensor(
            [list(map(lambda x: -100 if x == 0 else x, elem)) for elem in
             encoded_labels.tolist()]
        )
        training_arguments = TrainingArguments(
            output_dir=path,
            overwrite_output_dir=True,
            do_train=True,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            save_strategy='no',
            seed=seed,
            no_cuda=self.device == torch.device('cpu'),
            logging_strategy='epoch',
        )
        dataset = ProjectDataset(encoded_sentences, encoded_labels)
        self.trainer = Trainer(
            model=self.generator,
            args=training_arguments,
            train_dataset=dataset,

            callbacks=[CustomStoppingCallback(patience=2)
                       ] if early_stop else []
        )
        self.trainer.train()
        # Log losses at the end of the training
        if hasattr(self.trainer.state, 'log_history'):
            losses = [
                f"{l['loss']:.2f}" for l in self.trainer.state.log_history if
                l.get('loss')]
            logging.info(f"Losses: {losses}")

        if save:
            self.saveModel(path)

    def loadModel(self, path=None):
        path = self._definePath(path)

        logging.info(f"Loading T5 model from {path}")

        self.generator = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path, use_fast=True)

        if self.fp16:
            self.generator.half()

    def saveModel(self, path=None):
        path = self._definePath(path)

        if self.fp16:
            self.generator.half()

        self.tokenizer.save_vocabulary(path)
        self.generator.save_pretrained(path)

    def generate(
            self, sentences, max_length=None, min_length=None, num_beams=1,
            temperature=1.0, top_k=50, top_p=1, repetition_penalty=1.0,
    ):
        generations = []
        if isinstance(sentences, str):
            sentences = [sentences]
        for sentence in sentences:
            current_sentence_ids = self.tokenizer(
                sentence, return_tensors='pt').input_ids
            current_generation = self.generator.generate(
                current_sentence_ids, max_length=max_length,
                min_length=min_length, num_beams=num_beams,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty)[0]
            generations.append(
                self.tokenizer.decode(
                    current_generation, skip_special_tokens=True)
            )
        return generations
