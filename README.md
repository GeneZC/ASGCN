# ASGCN

**ASGCN** - **A**spect-**S**pecific **G**raph **C**onvolutional **N**etwork
* Code and preprocessed dataset for [EMNLP 2019](https://www.emnlp-ijcnlp2019.org/program/accepted/) paper titled "[Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks](https://dl.acm.org/citation.cfm?id=3331351)" 
* [Chen Zhang](https://genezc.github.io), [Qiuchi Li](https://qiuchili.github.io) and Dawei Song.

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
@inproceedings{Zhang,
 author = {Zhang, Chen and Li, Qiuchi and Song, Dawei},
 title = {Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks},
 series = {EMNLP'19},
 year = {2019},
 publisher = {ACL},
}
```

## Credits

* Code of this repo heavily relies on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch), in which I am one of the contributors.
* For any issues or suggestions about this work, don't hesitate to create an issue or directly contact me via [gene_zhangchen@163.com](mailto:gene_zhangchen@163.com) !