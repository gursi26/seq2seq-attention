# nmt-with-attention
A PyTorch implementation of "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau et al

[Paper](https://arxiv.org/abs/1409.0473) </br>
[Dataset Source](http://www.manythings.org/anki/) </br>
[GloVe embeddings](https://nlp.stanford.edu/projects/glove/)

Sample outputs after brief training:
```python
>>> generate_sample(encoder, decoder, dataset, "i like to play tennis", DEV)
'me gusta jugar al tenis'

>>> generate_sample(encoder, decoder, dataset, "i like to watch movies", DEV)
'me gusta ver películas al cine'

>>> generate_sample(encoder, decoder, dataset, "i have a car", DEV)
'tengo un coche de trabajo'
```
