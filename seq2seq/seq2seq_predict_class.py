from t5_generator import T5Generator
import ast
from tqdm import tqdm

dummy_output = ("", "")


class Seq2seqModelPredict:
    def __init__(self, trained_model):
        self.t5_model = T5Generator('google/t5-v1_1-small')
        self.t5_model.loadModel(trained_model)

    def _processGeneration(self, generation):
        splitted_gen = generation.split('; ')
        out_list = []
        current_tuple = []
        for elem in splitted_gen:
            try:
                current_tuple = ast.literal_eval(elem)
            except:
                pass
            if not isinstance(current_tuple, tuple):
                current_tuple = dummy_output
            try:
                dd = current_tuple[0]
                dd = current_tuple[1]
            except:
                current_tuple = dummy_output
            out_list.append(current_tuple)
        if not out_list:
            out_list.append(dummy_output)
        return out_list

    def predict(self, sentences):
        generations = self.t5_model.generate(sentences=sentences)
        predictions = [self._processGeneration(elem) for elem in tqdm(generations)]
        return predictions
