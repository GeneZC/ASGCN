# ASGCN

**ASGCN** - **A**spect-**S**pecific **G**raph **C**onvolutional **N**etwork
* Code and preprocessed dataset for [EMNLP 2019](https://www.emnlp-ijcnlp2019.org/program/accepted/) paper titled "[Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks](https://arxiv.org/abs/1909.03477)" 
* [Chen Zhang](https://genezc.github.io), [Qiuchi Li](https://qiuchili.github.io) and [Dawei Song](http://cs.bit.edu.cn/szdw/jsml/js/sdw/index.htm).

## Requirements

* Python 3.6
* PyTorch 1.0.0
* SpaCy 2.0.18
* numpy 1.15.4

## Usage

* Install [SpaCy](https://spacy.io/) package and language models with
```bash
pip install spacy
```
and
```bash
python -m spacy download en
```
* Generate graph data with
```bash
python dependency_graph.py
```
* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and extract `glove.840B.300d.txt` into `glove/`.
* Train with command, optional arguments could be found in [train.py](/train.py)
```bash
python train.py --model_name asgcn --dataset rest14 --save True
```
* Infer with [infer.py](/infer.py)

## Model

we propose to build a Graph Convolutional Network (GCN) over the dependency tree of a sentence to exploit syntactical information and word dependencies. Based on it, a novel aspectspecific sentiment classification framework is raised.

An overview of our proposed model is given below

![model](/assets/model.png)

## Citation

If you use the code in your paper, please kindly star this repo and cite our paper

```bibtex
@inproceedings{zhang-etal-2019-aspect, 
    title = "Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks", 
    author = "Zhang, Chen and Li, Qiuchi and Song, Dawei", 
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)", 
    month = nov, year = "2019", 
    address = "Hong Kong, China", 
    publisher = "Association for Computational Linguistics", 
    url = "https://www.aclweb.org/anthology/D19-1464", 
    doi = "10.18653/v1/D19-1464", 
    pages = "4560--4570",
} 
```

## Credits

* Code of this repo heavily relies on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch), in which I am one of the contributors.
* For any issues or suggestions about this work, don't hesitate to create an issue or directly contact me via [gene_zhangchen@163.com](mailto:gene_zhangchen@163.com) !