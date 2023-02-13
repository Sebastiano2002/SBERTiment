import numpy as np
import pickle
import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
    Trainer, TrainingArguments, TrainerCallback
import logging
from os.path import exists, join



class ProjectDataset(Dataset):
    def __init__(self, encodings, intents):
        self.encodings = encodings
        self.intents = intents

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.intents[idx])
        return item


class CustomStoppingCallback(TrainerCallback):
    def __init__(self, patience=1, refactor_ratio=0.95):
        self.patience = patience
        self.refactor_ratio = refactor_ratio
        self.patience_counter = 0
        self.best_loss = float('+inf')
        self.epochs = 0
        self.steps = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Run before every training epoch, it reads the loss value from the
        last log.
        In order to have every time the most recent values it requires to set
        logging_strategy='epoch' in TrainingArguments"""

        if state.log_history:
            loss = state.log_history[-1]['loss']
            self.epochs = state.log_history[-1]['epoch']
            self.steps = state.log_history[-1]['step']

            refactored_best_loss = self.best_loss * self.refactor_ratio
            if loss < refactored_best_loss:
                self.best_loss = loss
            else:
                logging.info(
                    f"{loss} <than refactored best loss {refactored_best_loss}")
                print(
                    f"{loss} <than refactored best loss {refactored_best_loss}")
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                control.should_training_stop = True

        return control


class BertClassifier:
    def __init__(self, pre_trained_model):
        self.pre_trained_model = pre_trained_model
        self.clf = None
        self.tokenizer = None
        self.trainer = None
        self.device = 'auto'
        self.lab2idx = {}
        self.idx2lab = {}
        self.fp16 = False

    def toFp16(self):
        self.fp16 = True
        if self.clf is not None:
            self.clf.half()

    @staticmethod
    def mapIntentToLabel(intents):
        uniqueIntents = set(intents)
        intentsToLabels = dict(
            zip(list(uniqueIntents), range(len(uniqueIntents))))
        return intentsToLabels

    def _encodeSentences(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        max_length = 128
        encoded_sentences = self.tokenizer(
            sentences, add_special_tokens=True, padding="max_length",
            truncation=True, max_length=max_length)

        if encoded_sentences.get("token_type_ids"):
            encoded_sentences.pop("token_type_ids")
        return encoded_sentences

    def _transformLabels(self, intents):
        self.intent2label = self.mapIntentToLabel(intents)
        self.label2intent = {v: k for k, v in self.intent2label.items()}
        transformed_labels = [self.intent2label[label] for label in intents]
        return transformed_labels

    def _definePath(self, path):
        if not path.endswith("/"):
            path += "/"
        return path

    def _setDevice(self, device=None):
        if device is not None and device != 'auto':
            self.device = torch.device(device)
            return

        if self.device == "auto":
            self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self.device = torch.device(self.device)

    def fit(
            self, sentences, intents, output_path, device='auto',
            per_device_train_batch_size=32, learning_rate=10e-5, epochs=10,
            seed=42, early_stop=False, save=True, newtrain=True
    ):
        path = self._definePath(output_path)

        self._setDevice(device)
        integer_intents = self._transformLabels(intents)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pre_trained_model, use_fast=True)
        if newtrain:
            self.clf = AutoModelForSequenceClassification.from_pretrained(
                self.pre_trained_model, num_labels=len(set(integer_intents)))
        else:
            self.clf = AutoModelForSequenceClassification.from_pretrained(
                self.pre_trained_model)
        self.clf.to(self.device)

        self.clf.train()

        encoded_sentences = self._encodeSentences(sentences)
        training_arguments = TrainingArguments(
            output_dir=output_path,
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
        dataset = ProjectDataset(encoded_sentences, integer_intents)
        self.trainer = Trainer(
            model=self.clf,
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

    def loadIntentMap(self, path=None, intent_map=None):
        """Load label2intent.pickle file in the model directory
        """
        if not intent_map:

            if exists(join(path, "label2intent.pickle")):
                filename = join(path, "label2intent.pickle")
            else:
                logging.error("Loading from lab2idx.pickle is deprecated")
                filename = join(path, "lab2idx.pickle")

            with open(filename, "rb") as infile:
                intent_map = pickle.load(infile)

        first_key = list(intent_map.keys())[0]

        if isinstance(first_key, int) or first_key.isdigit():
            self.label2intent = intent_map
            self.intent2label = {v: k
                                 for k, v in self.label2intent.items()}
        else:
            self.intent2label = intent_map
            self.label2intent = {v: k
                                 for k, v in self.intent2label.items()}

    def loadModel(self, path=None):
        path = self._definePath(path)

        logging.info(f"Loading BERT model from {path}")

        self.clf = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        self.loadIntentMap(path)

        if self.fp16:
            self.clf.half()

    def saveModel(self, path=None):
        path = self._definePath(path)

        if self.fp16:
            self.clf.half()

        self.tokenizer.save_vocabulary(path)
        self.clf.save_pretrained(path)

        with open(join(path, "label2intent.pickle"), "wb") as outfile:
            pickle.dump(self.label2intent, outfile)

    def predictSentences(self, sentences, batch_size=10):
        self._setDevice()
        encoded_sentences = self._encodeSentences(sentences)
        input_ids = torch.tensor(encoded_sentences["input_ids"])
        attention_masks = torch.tensor(encoded_sentences["attention_mask"])
        data = TensorDataset(input_ids, attention_masks)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(
            data, sampler=sampler, batch_size=batch_size)
        self.clf.eval()
        self.clf.to(self.device)
        predictions = []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            batch_input_ids, batch_masks = batch
            output = self.clf(
                batch_input_ids,
                attention_mask=batch_masks)
            logits = output[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
        return np.concatenate(predictions, axis=0)

    def predictBests(self, sentences):
        predictions = self.predictSentences(sentences)
        indices = np.argmax(predictions, axis=1).tolist()
        return [self.label2intent[i] for i in indices]
