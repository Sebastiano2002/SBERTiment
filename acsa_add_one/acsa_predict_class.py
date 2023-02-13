from bert_classifier import BertClassifier
from tqdm import tqdm


input_template = "{subtopic} [SEP] {text}"

class ACSAPredictor:
    def __init__(self, trained_model, categories_list):
        self.model = BertClassifier('bert-base-uncased')
        self.model.loadModel(trained_model)
        self.categories = categories_list

    def predictSentence(self, sentence):
        inputs = [input_template.format(subtopic=elem, text=sentence) for elem in self.categories]
        pred_labels = self.model.predictBests(inputs)
        admitted_categories = [self.categories[idx] for idx in range(len(self.categories)) if pred_labels[idx] != 'no']
        admitted_labels = [pred_labels[idx] for idx in range(len(pred_labels)) if pred_labels[idx] != 'no']
        if not admitted_categories:
            output = []
        else:
            output = [(admitted_categories[idx], admitted_labels[idx]) for idx in range(len(admitted_categories))]
        return output

    def predictSentences(self, sentences):
        return [self.predictSentence(sentence) for sentence in tqdm(sentences)]
