# AllenNLP Models

(tested in `Python 3.6` and `Python 3.7`)

This is a repository of NLP models using [AllenNLP](https://allennlp.org/).<br>
For each model, there is a simple example showing how to use it in practice.

Right now there are two models:
- a **Text Classification** model using:
  - randomly initialized **Word Embeddings**
  - a **CNN** encoder
- a **Sequence Labelling** model using:
  - pre-trained contextualized embeddings (**ELMo**)
  - a **Bi-LSTM** encoder
