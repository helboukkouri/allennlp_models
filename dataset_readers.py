from typing import Iterator, List, Dict, Optional
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader


class ClassificationDatasetReader(DatasetReader):
    """
    DatasetReader for Text Classification.
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.tokenizer = WordTokenizer()

    @overrides
    def text_to_instance(self, tokens: List[Token], label: str) -> Instance:
        # pylint: disable=arguments-differ
        fields = {
            "tokens": TextField(tokens, token_indexers=self.token_indexers),
            "label": LabelField(label)
        }
        return Instance(fields)

    @overrides
    def read(self, path: str) -> Iterator[Instance]:
        # pylint: disable=arguments-differ
        data = []
        with open(path, 'r') as f:
            for line in f:
                splitline = line.strip().split()
                label = splitline[0]
                text = ' '.join(splitline[1:])
                tokens = [str(t) for t in self.tokenizer.tokenize(text)]
                assert '__label__' in label
                label = label.split('__label__')[-1]
                data.append(self.text_to_instance([Token(t) for t in tokens], label))
        return data


class SequenceLabellingDatasetReader(DatasetReader):
    """
    DatasetReader for Sequence Labelling.
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.tokenizer = WordTokenizer()

    @overrides
    def text_to_instance(self, tokens: List[Token], labels: List[str]) -> Instance:
        # pylint: disable=arguments-differ
        tokens_field = TextField(tokens, token_indexers=self.token_indexers)
        fields = {
            "tokens": tokens_field,
            "labels": SequenceLabelField(labels, sequence_field=tokens_field)
        }
        return Instance(fields)

    @overrides
    def read(self, path: str) -> Iterator[Instance]:
        # pylint: disable=arguments-differ
        data = []
        with open(path, 'r') as f:
            tokens, labels = [], []
            for line in f:
                splitline = line.strip().split()
                if splitline:
                    assert len(splitline) == 2
                    token = splitline[0].lower()
                    tokens.append(token)
                    label = splitline[1]
                    labels.append(label)
                else:
                    data.append(self.text_to_instance([Token(t) for t in tokens], labels))
                    tokens, labels = [], []
        return data
