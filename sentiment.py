from transformers import BertModel, BertConfig, BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, BertForPreTraining
import windowing
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.special import softmax

#Reading the latest saved model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
config = BertConfig.from_json_file('model/config.json')
model = BertForSequenceClassification.from_pretrained('model/pytorch_model.bin', config=config)


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def sentiment_analysis(texts):
    texts = texts, "" #For Trainer to see the text, dataset atleast needs to have 2 elements
    tokenized_texts = tokenizer(texts, padding=True)

    test_dataset = SimpleDataset(tokenized_texts)

    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    return softmax(predictions.predictions[0]), predictions.predictions[0]


if __name__ == "__main__":
    texts = "I  understand general complex exponential signals part of the video very well."
    print(sentiment_analysis(texts))
    window_length = 6
    chapter_id = 3
    video_id = 1
    if sentiment_analysis(texts)[0][0] > sentiment_analysis(texts)[0][1]:
        matched_window_text, sim_val, idx = windowing.cos_sim_calc(texts, chapter_id, video_id, window_length)
        print("Input test is labelled negative for the Chapter {} video {} and the matched texts ".format(chapter_id, video_id))
        print("matched texts with cosine similarity of {}:".format(sim_val))
        print(matched_window_text)
    else:
        print("Input test is labelled positive for the Chapter {} video {}.".format(chapter_id,video_id))