from typing import Iterator, List, Dict, Optional

import numpy
import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure

class TextClassifier(Model):
    """
    This Model performs text classification.
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder) -> None:

        super(TextClassifier, self).__init__(vocab=vocabulary)
        
        self.vocabulary = vocabulary
        self.embedder = embedder
        self.encoder = encoder

        self.num_classes = self.vocabulary.get_vocab_size("labels")

        self.feedforward = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=self.num_classes)

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self.loss = torch.nn.CrossEntropyLoss()


    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        embeddings = self.embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        encoder_output = self.encoder(embeddings, mask)

        logits = self.feedforward(encoder_output)

        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            output_dict["loss"] = loss
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))

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
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {name: metric.get_metric(reset) for name, metric in self.metrics.items()}


class SequenceLabeller(Model):
    """
    This Model performs text classification.
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:

        super(SequenceLabeller, self).__init__(vocab=vocabulary)
        
        self.vocabulary = vocabulary
        self.embedder = embedder
        self.encoder = encoder

        self.num_classes = self.vocabulary.get_vocab_size("labels")

        self.feedforward = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=self.num_classes)

        self.accuracy = CategoricalAccuracy()
        self.F1 = SpanBasedF1Measure(vocabulary, tag_namespace='labels')
        self.loss = util.sequence_cross_entropy_with_logits


    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        embeddings = self.embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        encoder_output = self.encoder(embeddings, mask)

        logits = self.feedforward(encoder_output)

        output_dict = {'logits': logits}
        if labels is not None:
            loss = self.loss(logits, labels, mask)
            output_dict["loss"] = loss
            self.accuracy(logits, labels, mask)
            self.F1(logits, labels, mask)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"labels"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [
            [self.vocab.get_token_from_index(i, namespace="labels") for i in indices]
            for indices in argmax_indices]
        output_dict['labels'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {"accuracy": self.accuracy.get_metric(reset)}

        for name, value in self.F1.get_metric(reset).items():
            if 'overall' in name:
                metrics[name] = value

        return metrics
