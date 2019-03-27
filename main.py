import warnings

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator

import torch
import torch.optim as optim
from torch.nn import LSTM

from sklearn.metrics import classification_report

from dataset_readers import ClassificationDatasetReader, SequenceLabellingDatasetReader
from models import TextClassifier, SequenceLabeller


EMBEDDING_DIM = 100
NUM_FILTERS = 32
FILTER_SIZES = (2, 3)
LEARNING_RATE = 0.001
BATCH_SIZE = 2
NUM_EPOCHS = 10
HIDDEN_SIZE = 32

# "Small" ELMO (embedding size = 2 * 128)
ELMO_OPTIONS_FILE = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                     '/2x1024_128_2048cnn_1xhighway/'
                     'elmo_2x1024_128_2048cnn_1xhighway_options.json')
ELMO_WEIGHTS_FILE = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                     '/2x1024_128_2048cnn_1xhighway/'
                     'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
ELMO_EMBEDDING_DIM = 2 * 128

warnings.filterwarnings(action='ignore')


def train_model(model, training_data, validation_data, vocabulary):
    
    # Use Adam from optimization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create data batches
    iterator = BucketIterator(
        batch_size=BATCH_SIZE,
        sorting_keys=[("tokens", "num_tokens")])  # Sort by sequence lenght to optimize padding
    iterator.index_with(vocabulary)
    
    # Train with early stopping and GPU if available
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=training_data,
        validation_dataset=validation_data,
        patience=3, num_epochs=NUM_EPOCHS,
        cuda_device=-1)  # Can be changed to 0 or 1 to use the 1st or 2nd visible cuda device
    trainer.train()


def evaluate_classification_model(model, test_data):
    y_true = []
    test_instances = []
    for instance in test_data:
        test_instances.append(instance)
        y_true.append(instance["label"].label)

    y_pred = [pred["label"] for pred in model.forward_on_instances(test_instances)]

    print("\nClassification Report :\n")
    print(classification_report(y_true, y_pred))


def evaluate_sequence_labelling_model(model, test_data):
    y_true = []
    test_instances = []
    for instance in test_data:
        test_instances.append(instance)
        y_true.extend(instance["labels"].labels)

    y_pred = [
        label 
        for pred in model.forward_on_instances(test_instances)
        for label in pred["labels"]]

    print("\nClassification Report :\n")
    print(classification_report(y_true, y_pred))


def classification():

    # Index each token with a single Id
    token_indexers = {"tokens": SingleIdTokenIndexer()}

    # Read the data
    reader = ClassificationDatasetReader(token_indexers)
    training_data = reader.read(path='data/classification/train.txt')
    validation_data = reader.read(path='data/classification/test.txt')
    test_data = reader.read(path='data/classification/test.txt')

    # Create a vocabulary
    vocabulary = Vocabulary.from_instances(training_data + validation_data + test_data)

    # Create an "Embedder" from a randomly initialized embedding layer
    embedding_layer = torch.nn.Embedding(
        num_embeddings=vocabulary.get_vocab_size('tokens'),
        embedding_dim=EMBEDDING_DIM)
    embedder = BasicTextFieldEmbedder(
        token_embedders={"tokens": embedding_layer})

    # Our text classifier will use a CNN encoder
    cnn_encoder = CnnEncoder(
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        ngram_filter_sizes=FILTER_SIZES,
        output_dim=NUM_FILTERS * len(FILTER_SIZES))
    model = TextClassifier(
        vocabulary=vocabulary,
        embedder=embedder,
        encoder=cnn_encoder)

    print("\nModel :\n")
    print(model)

    # Training
    train_model(model, training_data, validation_data, vocabulary)

    # Evaluation
    evaluate_classification_model(model, test_data)


def sequence_labelling():

    # Index each token as a sequence of character Ids (ELMo)
    token_indexers = {"tokens": ELMoTokenCharactersIndexer()}

    # Read the data
    reader = SequenceLabellingDatasetReader(token_indexers)
    training_data = reader.read(path='data/sequence_labelling/train.txt')
    validation_data = reader.read(path='data/sequence_labelling/test.txt')
    test_data = reader.read(path='data/sequence_labelling/test.txt')

    # Create a vocabulary
    vocabulary = Vocabulary.from_instances(training_data + validation_data + test_data)

    # Use ELMo embeddings
    elmo = ElmoTokenEmbedder(
        options_file=ELMO_OPTIONS_FILE, weight_file=ELMO_WEIGHTS_FILE)

    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": elmo})

    # Our text classifier will use a CNN encoder

    lstm_layer = LSTM(
        input_size=ELMO_EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        bidirectional=True,
        batch_first=True)
    lstm_encoder = PytorchSeq2SeqWrapper(module=lstm_layer)

    model = SequenceLabeller(
        vocabulary=vocabulary,
        embedder=embedder,
        encoder=lstm_encoder)

    print("\nModel :\n")
    print(model)

    # Training
    train_model(model, training_data, validation_data, vocabulary)

    # Evaluation
    evaluate_sequence_labelling_model(model, test_data)


if __name__ == "__main__":
    print("=====================================")
    print("Running a classification example ...")
    print("=====================================\n")
    classification()

    print("=========================================")
    print("Running a sequence labelling example ...")
    print("=========================================\n")
    sequence_labelling()
