from typing import Iterator, List, Dict, Optional

import numpy
import torch
import torch.optim as optim
import torch.nn.functional as F
from overrides import overrides
from sklearn.metrics import classification_report

from allennlp.nn import util
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer


class TextClassifier(Model):
    """
    This Model performs text classification.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TextClassifier, self).__init__(vocab, regularizer)

        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.feedforward = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=self.num_classes)

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        embedded_text = self.text_field_embedder(tokens)
        text_mask = util.get_text_field_mask(tokens)

        encoded_text = self.encoder(embedded_text, text_mask)

        logits = self.feedforward(encoded_text)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'])
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {name: metric.get_metric(reset) for name, metric in self.metrics.items()}


class ClassificationDatasetReader(DatasetReader):
    """
    DatasetReader for text Classification.
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.tokenizer = WordTokenizer()

    @overrides
    def text_to_instance(self, tokens: List[Token], label: str) -> Instance:
        # pylint: disable=arguments-differ
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        label_field = LabelField(label)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, path: str) -> Iterator[Instance]:
        # pylint: disable=arguments-differ
        with open(path, 'r') as f:
            for line in f:
                splitline = line.strip().split()
                label = splitline[0]
                text = ' '.join(splitline[1:])
                tokens = [str(t) for t in self.tokenizer.tokenize(text)]
                assert '__label__' in label
                label = label.split('__label__')[-1]
                yield self.text_to_instance([Token(t) for t in tokens], label)


if __name__ == "__main__":

    reader = ClassificationDatasetReader(
        token_indexers={"tokens": SingleIdTokenIndexer()})
    training_data = reader._read(path='data/classification/train.txt')
    test_data = reader._read(path='data/classification/test.txt')

    vocabulary = Vocabulary.from_instances(training_data)

    EMBEDDING_DIM = 100
    embedding_layer = torch.nn.Embedding(
        num_embeddings=vocabulary.get_vocab_size('tokens'),
        embedding_dim=EMBEDDING_DIM)

    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding_layer})

    NUM_FILTERS = 32
    FILTER_SIZES = (2, 3)
    cnn = CnnEncoder(
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        ngram_filter_sizes=FILTER_SIZES,
        output_dim=NUM_FILTERS * len(FILTER_SIZES))

    model = TextClassifier(
        vocab=vocabulary, text_field_embedder=embedder, encoder=cnn)

    LEARNING_RATE = 0.001
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    BATCH_SIZE = 2
    iterator = BucketIterator(
        batch_size=BATCH_SIZE,
        sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocabulary)
    
    NUM_EPOCHS = 10
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=training_data,
                      validation_dataset=test_data,
                      patience=3,
                      num_epochs=NUM_EPOCHS,
                      cuda_device=0 if torch.cuda.is_available() else -1)

    trainer.train()
    
    y_true = []
    test_instances = []
    for instance in reader.read('data/classification/test.txt'):
        test_instances.append(instance)
        y_true.append(instance["label"].label)

    y_pred = [pred["label"] for pred in model.forward_on_instances(test_instances)]

    print(classification_report(y_true, y_pred))
