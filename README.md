Official implementation of [Exploiting Entity BIO Tag Embeddings and Multi-task Learning for Relation Extraction with Imbalanced Data](https://www.aclweb.org/anthology/P19-1130/).

Contributed by National Engineering Research Center for Software Engineering, Peking University.

## Overview

This is a TensorFlow-based framwork for Relation Extraction using BIO tag embeddings and multi-task learning, we use Keras to easily implement our methods. Our model has three parts:

- Input Layer:  
	- Word Embeddings
	- Positional Embeddings
	- BIO tag Embeddings
- Convolutional Layer with Multi-Sized Window Kernels
- Multi-Task Layer: 
	- Relation Identification with Cross-entropy Loss
	- Relation Classification with Ranking Loss

## Data Generation and Parameter Settings

The dataset we used in this paper is AEC2005 (English and Chinese corpus), which is a very popular dataset for relation extraction. The Data Preparation and Parameter Settings are mentioned in our paper, we will also release the processed data later to facilitate future research.

## Requirements

- Python(3.6)
- Numpy(>=1.13.3)
- Tensorflow (>=1.9)
- Keras(>=2.1.1)
- scikit-learn(>=0.18)

## Test Results

### English Corpus:

|Model | P% | R% | F1% |
|  ----  | ----  | ----  | ----  |
|SPTree | 70.1 | 61.2 | 65.3 |
|Walk-based | 69.7 | 59.5 | 64.2 |
|Baseline | 58.8 | 57.3 | 57.2 |
|Baseline+Tag | 61.3 | 76.7 | 67.4 |
|Baseline+MTL | 63.8 | 56.1 | 59.5 |
|Baseline+MTL+Tag | 66.5 | 71.8 | 68.9 |

### Chinese Corpus:

| Model | P% | R% | F1% |
| ---- | ---- | ---- | ---- |
| PCNN | 54.4 | 42.1 | 46.1 |
| Eatt-BiGRU | 57.8 | 49.7 | 52.0 |
| Baseline | 48.5 | 57.1 | 51.7 |
| Baseline+Tag | 61.8 | 62.7 | 61.4 |
| Baseline+MTL | 56.7 | 52.9 | 53.8 |
| Baseline+MTL+Tag | 61.3 | 65.8 | 62.9 |


## Citation

```
@inproceedings{ye-etal-2019-exploiting,
    title = "Exploiting Entity {BIO} Tag Embeddings and Multi-task Learning for Relation Extraction with Imbalanced Data",
    author = "Ye, Wei  and
      Li, Bo  and
      Xie, Rui  and
      Sheng, Zhonghao  and
      Chen, Long  and
      Zhang, Shikun",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1130",
    doi = "10.18653/v1/P19-1130",
    pages = "1351--1360"
}
```