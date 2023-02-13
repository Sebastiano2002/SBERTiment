from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from aspect_extractor_class import Absa
from tqdm import tqdm


class AbsaPredictPipelineSiamese:
    def __init__(self, absa_model, topic_match_model, topics_list):

        self.absa_model = Absa(project='absa_paper')
        self.absa_model_name = absa_model
        self.topic_matcher = SentenceTransformer(topic_match_model)
        self.topic_list = topics_list
        self.topics_embeddings = self.topic_matcher.encode(topics_list)

    def semanticSearchPredictor(self, queries):
        query_encoding = self.topic_matcher.encode(queries)
        out = semantic_search(query_encoding, self.topics_embeddings, top_k=1)
        return [self.topic_list[out[idx][0]['corpus_id']] for idx in range(len(queries))]


    def predict(self, sentences):
        final_predictions = []
        predictions_absa = self.absa_model.predict(sentences, model_name=self.absa_model_name)
        for idx, pred in tqdm(enumerate(predictions_absa)):
            current_aspects = pred.aspects
            if not current_aspects:
                to_append = []

            else:
                curr_labels = [elem.label for elem in current_aspects]
                curr_aspect_words = [elem.word for elem in current_aspects]
                curr_input = [f"{sentences[idx]} [SEP] {curr_aspect_words[idx2]}" for idx2 in range(len(curr_aspect_words))]
                topics_predicted = self.semanticSearchPredictor(curr_input)
                to_append = [(topics_predicted[idx3], curr_aspect_words[idx3], curr_labels[idx3]) for idx3 in range(len(topics_predicted))]
            final_predictions.append(to_append)
        return final_predictions
