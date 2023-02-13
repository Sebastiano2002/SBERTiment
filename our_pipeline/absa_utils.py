import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, TrainerCallback

logger = logging.getLogger(__name__)


@dataclass
class AbsaBertParams:
    pre_trained_model: str = "bert-base-uncased"
    batch_size: int = 16
    learning_rate: float = 5e-5
    epochs: int = 7
    seed: int = 42
    device: str = "auto"
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    logging_steps: int = 50


@dataclass
class Aspect:
    word: str
    label: str
    confidence: float
    range_begin: int
    range_end: int


@dataclass
class Prediction:
    words: List[str]
    labels: List[str]
    aspects: List[Aspect]


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence.
                This should be specified for train and dev examples, but not
                for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]
    text: str


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


@dataclass
class SeqInputFeatures:
    """A single set of features of data for the ABSA task"""
    input_ids: list
    input_mask: list
    segment_ids: list
    label_ids: list
    evaluate_label_ids: list


class CustomStoppingCallback(TrainerCallback):
    def __init__(self, patience=1):
        self.patience = patience

        # patience_counter denotes the number of times validation metrics
        # failed to improve.
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

            refactored_best_loss = self.best_loss * 0.95
            if loss < refactored_best_loss:
                self.best_loss = loss
            else:
                logging.info(f"{loss} <than refactored best loss "
                             "{refactored_best_loss}")
                print(f"{loss} <than refactored best loss "
                      "{refactored_best_loss}")
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                control.should_training_stop = True

        return control


class TokenClassificationTask:
    @staticmethod
    def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ) -> List[InputFeatures]:
        """Loads a data file into a list of `InputFeatures`
        `cls_token_segment_id` define the segment id associated to the CLS token
        (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10_000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            tokens = []
            label_ids = []
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([])
                # when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word,
                    # and padding ids for the remaining tokens
                    label_ids.extend([label_map[label]] + [pad_token_label_id]
                                     * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=input_mask,
                    token_type_ids=segment_ids, label_ids=label_ids
                )
            )
        return features


class TokenClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        token_classification_task: TokenClassificationTask,
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
    ):

        self.features = token_classification_task.convert_examples_to_features(
            examples,
            labels,
            max_seq_length,
            tokenizer,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            # roberta uses an extra separator b/w pairs of sentences
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def fromWordsToAspects(words, sentiments, confidences=None):
    if confidences is None:
        confidences = [1]*len(words)
    aspects = []

    local_aspect = ''
    local_sent = 'O'
    local_cfs = []

    text = ''
    aspect_begin = 0
    aspect_end = 0

    for word, s, cf in zip(words, sentiments, confidences):
        if s in ['T-POS', 'T-NEU', 'T-NEG']:
            local_cfs.append(cf)

            if local_aspect:
                # FIXME? stiamo assumendo che due parole consecutive se
                # entrambe taggate abbiano lo stesso sentiment
                local_aspect = local_aspect + ' ' + word
                aspect_end += 1 + len(word)
            else:
                local_sent = s
                local_aspect = word

                if text:
                    aspect_begin = len(text) + 1  # +1 is for the space
                aspect_end = aspect_begin + len(word)

        else:
            if local_aspect:
                aspects.append(Aspect(
                    word=local_aspect, label=local_sent,
                    confidence=np.mean(local_cfs),
                    range_begin=aspect_begin,
                    range_end=aspect_end
                ))

                local_aspect = ''
                local_cfs = []
                aspect_begin = 0
                aspect_end = 0

        if text:
            text = text + ' ' + word
        else:
            text = word

    if local_aspect:
        aspects.append(Aspect(
            word=local_aspect, label=local_sent,
            confidence=np.mean(local_cfs),
            range_begin=aspect_begin,
            range_end=aspect_end
        ))

    return aspects



def computeAbsaMetrics(examples, predictions, misc_path=None):
    """
    examples: list of InputExample
    predictions: list of (words, labels, Aspect)
    """
    shallow_acc = 0
    tp = 0
    fp = 0
    fn = 0
    n_pred_aspetti = 0
    tot_pred_aspetti = 0
    prec_sentiment = 0
    tot_true_aspetti = 0
    n_pred_aspetti_included = 0
    df = pd.DataFrame(
        columns=['sentence', 'true_output', 'pred_output', 'isCorrect']
    )

    for idx in range(len(examples)):
        ex = examples[idx]
        pred = predictions[idx]
        assert len(pred.labels) == len(ex.labels)
        pred_output = fromWordsToAspects(pred.words, pred.labels)
        true_output = fromWordsToAspects(pred.words, ex.labels)

        true_output_dict = {x.word: x.label for x in true_output}
        pred_output_dict = {x.word: x.label for x in pred_output}
        for p in pred_output_dict.keys():
            tot_pred_aspetti += 1
            if p in true_output_dict.keys():
                n_pred_aspetti += 1
                if pred_output_dict[p] == true_output_dict[p]:
                    prec_sentiment += 1
            for k in true_output_dict.keys():
                if p in k:
                    n_pred_aspetti_included += 1
                    break

        tot_true_aspetti += len(true_output_dict)

        if pred.labels == list(ex.labels):
            if pred_output:
                tp += 1
            correct = True
            shallow_acc += 1
        else:
            if true_output:
                fn += 1
            if pred_output:
                fp += 1

            correct = False

        df = pd.concat(
            [df,
             pd.DataFrame({
                'sentence': [ex.text],
                'true_output': [true_output],
                'pred_output': [pred_output],
                'isCorrect': [correct]
             })],
            ignore_index=True)

    if misc_path:
        df.to_csv(misc_path, index=False)

    def percent(x):
        return round(100 * x, 2)

    prec = tp / (tp+fp) if (tp+fp) else 0.0
    recall = tp / (tp+fn) if (tp+fn) else 0.0
    if prec == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = round(2 / (1 / prec + 1 / recall), 2)

    return {
        "shallow_accuracy": percent(shallow_acc / len(examples)),
        "precision": percent(prec),
        "recall": percent(recall),
        "f1": percent(f1),
        "aspect_precision": percent(n_pred_aspetti / tot_pred_aspetti
                                    if tot_pred_aspetti else 0.0),
        "sentiment_precision": percent(prec_sentiment / n_pred_aspetti
                                       if n_pred_aspetti else 0.0),
        "aspect_recall": percent(n_pred_aspetti / tot_true_aspetti
                                 if tot_true_aspetti else 0.0),
        "aspects_precision_included": percent(n_pred_aspetti_included /
                                              tot_pred_aspetti
                                              if tot_pred_aspetti else 0.0)
    }


def firstClean(text):
    text = text.lower()
    text = re.sub(r"…", " . ", text)
    text = re.sub(r"\.{2,}", " . ", text)
    text = re.sub(r"\’", " ' ", text)
    subs = {
        r"l ?\'": "la ",
        r"un ?\'": "una ",
        r"app\.to": "appartamento",
        r"(^|\b|\s)(gg)($|\b|\s)": " giorni ",
        r"(\d)gg": "\g<1> giorni"
    }
    for pattern, replacement in subs.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\'", " ", text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def splitWords(text, tokenizer):
    """SPlit words and puntcation using the given tokenizer"""
    # Basic cleaning to avoid [UNK] tokens
    for quote in u"‘’´`":
        text = text.replace(quote, "'")
    for quote in u'“”':
        text = text.replace(quote, '"')
    text = text.replace("…", '...')

    words = []
    tokens = tokenizer.tokenize(text)

    for tk in tokens:
        if words and tk.startswith('##'):
            words[-1] += tk[2:]
        elif tk == '[UNK]':  # ignore unknown tokens
            pass
        else:
            words.append(tk)
    return words


def dfToExamples(df, tokenizer, n_aspects=4, lang_name=None):
    """Read excamples from a pandas dataframe"""

    def aspectIncluded(idx, text_list, asp_list):
        for n in range(len(asp_list) - 1):
            if text_list[idx+n+1] != asp_list[n+1]:
                return False

        return True

    examples = []
    errors_in_aspects = False
    for index, row in list(df.iterrows()):
        text = row['survey']

        if lang_name == 'it':
            text = firstClean(text)

        keywords = [row[f'aspect{i}'] for i in range(1, n_aspects + 1) if
                    not pd.isna(row[f'aspect{i}'])]
        sentiments = [row[f'sentiment{i}'] for i in range(1, n_aspects + 1) if
                      not pd.isna(row[f'sentiment{i}'])]

        aspects = {}
        for kw, sent in zip(keywords, sentiments):
            if lang_name == 'it':
                kw = firstClean(kw)

            aspects[kw] = sent

        words = splitWords(text, tokenizer)

        tags = ['O'] * len(words)

        for asp, sent in aspects.items():
            sent = f"T-{sent}" if sent in ['NEG', 'NEU', 'POS'] else sent
            asp_words = splitWords(asp, tokenizer)

            # Ignore aspects made by unknown tokens
            if not asp_words:
                continue

            indices = [i for i, w in enumerate(words) if w == asp_words[0]]
            check = False
            for idx_word in indices:
                if aspectIncluded(idx_word, words, asp_words):
                    start_idx = idx_word
                    end_idx = idx_word+len(asp_words)-1
                    tags[start_idx:end_idx+1] = [sent]*len(asp_words)
                    check = True
                    break
            if not check:
                errors_in_aspects = True
                logger.error(f"Unmatch in row {index} between\n"
                             f"Aspect:\t{asp}\nText:\t{text}")

        text = ' '.join(words)
        examples.append(InputExample(guid=str(index), text=text,
                                     labels=tags, words=words))

    if errors_in_aspects:
        raise ValueError("Unmatches between aspects and text")

    return examples


def txtToExamples(filename):
    """Read examples from a .txt file formatted using the OT schema"""
    examples = []

    with open(filename, 'r', encoding='UTF-8') as fp:
        sample_id = 0
        for index, line in enumerate(fp):
            sent_string, tag_string = line.strip().split('####')
            words = []
            tags = []
            for tag_item in tag_string.split(' '):
                eles = tag_item.split('=')
                if len(eles) == 1:
                    print(tag_string.split(' '))
                    raise Exception("Invalid samples %s..." % tag_item)
                elif len(eles) == 2:
                    word, tag = eles
                else:
                    word = ''.join((len(eles) - 2) * ['='])
                    tag = eles[-1]
                words.append(word)
                tags.append(tag)

            text = ' '.join(words)

            examples.append(InputExample(guid=str(index), text=text,
                                         labels=tags, words=words))
            sample_id += 1

    return examples


def platformToExamples(df, tokenizer):
    def readRange(r, sent):
        "range from string [a,b) to tuple (int, int, sentiment)"
        trimmed = r[1:-1]
        splitted = trimmed.split(',')
        return tuple(map(int, splitted)) + ('T-' + sent[:3],)

    def isAlnum(s):
        return s.isalnum() or s in ['€', '$']

    examples = []

    sample_ids = set(df.sample_id.tolist())

    for sample_id in sample_ids:
        sub_df = df[df.sample_id == sample_id]
        text = sub_df.text.tolist()[0]

        expected_n_words = len(splitWords(text, tokenizer))

        ranges = [readRange(r, sent)
                  for r, sent in zip(sub_df.range.tolist(),
                                     sub_df.sentiment.tolist())
                  if r != 'empty'
                  ]
        ranges = sorted(ranges, key=lambda r: r[0])
        # TODO: assert there are no overlapping ranges

        chunks = []  # tuples (slice of text, sentiment)
        chunk_begin = 0
        for (begin, end, sent) in ranges:

            # Fix offsets
            while begin > 0 and isAlnum(text[begin-1]):
                begin -= 1
                end -= 1

            # exclude trailing whitespaces or punct from aspect
            while end > begin and end < len(text) and not isAlnum(text[end-1]):
                end -= 1

            while end < len(text) and isAlnum(text[end]):
                end += 1

            if begin > chunk_begin:
                chunks.append((text[chunk_begin:begin], 'O'))

            chunks.append((text[begin:end], sent))
            chunk_begin = end
        chunks.append((text[chunk_begin:], 'O'))

        word_sents = []  # tuples (word, sentiment)
        for ch in chunks:
            text_slice, sent = ch
            if text_slice:
                words = splitWords(text_slice, tokenizer)
                sents = [sent] * len(words)
                word_sents.extend(zip(words, sents))

        words, sents = zip(*word_sents)
        text_joined = ' '.join(words)

        if not len(words) == expected_n_words:
            # Debug print, replace with assert
            print(chunks)
            print()

        # assert len(words) == expected_n_words, "\n{}\n\n{}".format(
        #     splitWords(text, tokenizer), words
        # )

        example = InputExample(guid=sample_id, words=words,
                               labels=sents, text=text_joined)
        examples.append(example)

    return examples
