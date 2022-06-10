import gzip
import json
import logging

import gensim
import spacy
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MonitorCallback(CallbackAny2Vec):
    def __init__(self):
        pass

    def on_epoch_end(self, model):
        print("Model loss:", model.get_latest_training_loss())  # print loss
        evaluate(model)


def read_file(path):
    with gzip.open(path) as f:
        data = f.readlines()
        js_data = json.loads(data[0])
        for img_desc in tqdm(js_data, desc="Loading dataset"):
            for desc in img_desc['regions']:
                yield gensim.utils.simple_preprocess(desc["phrase"])


def train():
    # Load doc
    doc = list(read_file("dataset/region_descriptions.json.gz"))

    model = gensim.models.Word2Vec(doc, size=300, window=10,
                                   min_count=2, workers=10)
    model.train(doc, total_examples=len(doc), epochs=5)
    model.save("word2vec_genome_skip_gram.model")
    model.wv.save_word2vec_format("word2vec.txt")
    return model


def evaluate(model=None):
    if model is None:
        model = gensim.models.Word2Vec.load("word2vec_genome.model")
    nlp = spacy.load('en_core_web_lg')

    words_pair = [
        ["knife", "fork"],
        ["cup", "bowl"],
        ['microwave', 'fridge'],
        ['sink', 'soap']
    ]
    for w_p in words_pair:
        print("Words", w_p[0], w_p[1])
        print("Visual genome", model.wv.similarity(w_p[0], w_p[1]))
        word1, word2 = nlp(w_p[0]), nlp(w_p[1])
        print("Spacy en_core_web_lg", word1.similarity(word2))
        print("")


def main():
    model = None
    model = train()
    evaluate(model)


if __name__ == '__main__':
    main()
