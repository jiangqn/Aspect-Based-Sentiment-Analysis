# Aspect-Base-Sentiment-Analysis

A paper list for aspect based sentiment analysis

## Datasets

## Paper List

### Aspect Base Sentiment Classification

#### RNN based model

- **TD_LSTM**: Effective LSTMs for Target-Dependent Sentiment Classification. COLING 2016. [[paper]][8] [[code]][21]

#### CNN based model

- **GCAE**: Aspect Based Sentiment Analysis with Gated Convolutional Networks. ACL 2018. [[paper]][2] [[code]][22]

#### Attention based model

- **MemNet**: Aspect Level Sentiment Classification with Deep Memory Network. EMNLP 2016. [[paper]][6] [[code]][23]

- **DAuM**: Enhanced Aspect Level Sentiment Classification with Auxiliary Memory. COLING 2018. [[paper]][11] [[code]][24]

- **AEN**: Attentional Encoder Network for Targeted Sentiment Classification. arXiv preprint 2019. [[paper]][28] [[code]][29]

#### RNN and Attention based model

- **ATAE_LSTM**: Attention-based LSTM for Aspect-level Sentiment Classification. EMNLP 2016. [[paper]][10] [[code]][25]

- **IAN**: Interactive Attention Networks for Aspect-Level Sentiment Classification. IJCAI 2017. [[paper]][1] [[code]][26]

- Effective Attention Modeling for Aspect-Level Sentiment Classification. COLING 2018. [[paper]][3]

- **MGAN**: Multi-grained Attention Network for Aspect-LevelSentiment Classification. ACL 2018. [[paper]][30] [[code]][31]

#### RecursiveNN based model

- Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification. ACL 2014. [[paper]][4]

- PhraseRNN: Phrase Recursive Neural Network for Aspect-based Sentiment Analysis. EMNLP 2015. [[paper]][7]

#### BERT base model

- **AEN_BERT / BERT_SPC**: Attentional Encoder Network for Targeted Sentiment Classification. arXiv preprint 2019. [[paper]][28] [[code]][29]

#### Transfer learning model

- **MGAN**: Exploiting Coarse-to-Fine Task Transfer for Aspect-level Sentiment Classification. AAAI 2019. [[paper]][12] [[data]][27]

- Exploiting Document Knowledge for Aspect-level Sentiment Classification. ACL 2018. [[paper]][9]

#### Hierarchical model

- Aspect Specific Sentiment Analysis using Hierarchical Deep Learning. NIPS 2014 workshop. [[paper]][5]

- A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis. EMNLP 2016. [[paper]][15]

#### Other

- Targeted Aspect-Based Sentiment Analysis via Embedding Commonsense Knowledge into an Attentive LSTM. AAAI 2018. [[paper]][13]

- Target-Sensitive Memory Networks for Aspect Sentiment Classification. ACL 2018. [[paper]][14]

- Modeling Inter-Aspect Dependencies for Aspect-Based Sentiment Analysis. NAACL 2018. [[paper]][16]

### Aspect Extraction

- An Unsupervised Neural Attention Model for Aspect Extraction. ACL 2017. [[paper]][17]

- **DECNN**: Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction. ACL 2018. [[paper]][18]

- Aspect Term Extraction with History Attention and Selective Transformation. IJCAI 2018. [[paper]][19]

- Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms. AAAI 2017. [[paper]][20]

[1]: https://arxiv.org/abs/1709.00893
[2]: https://arxiv.org/abs/1805.07043v1
[3]: http://aclweb.org/anthology/C18-1096
[4]: http://aclweb.org/anthology/P/P14/P14-2009.pdf
[5]: https://pdfs.semanticscholar.org/4500/68221da8297ac0a0e1524b1e196900c61b2e.pdf
[6]: https://arxiv.org/abs/1605.08900
[7]: http://www.aclweb.org/anthology/D15-1298
[8]: http://aclweb.org/anthology/C/C16/C16-1311.pdf
[9]: https://arxiv.org/abs/1806.04346
[10]: http://www.aclweb.org/anthology/D16-1058
[11]: http://aclweb.org/anthology/C18-1092
[12]: https://arxiv.org/abs/1811.10999
[13]: http://www.sentic.net/sentic-lstm.pdf
[14]: http://www.aclweb.org/anthology/P18-1088
[15]: https://arxiv.org/abs/1609.02745
[16]: http://www.aclweb.org/anthology/N18-2043
[17]: http://aclweb.org/anthology/P/P17/P17-1036.pdf
[18]: https://arxiv.org/abs/1805.04601v1
[19]: https://arxiv.org/abs/1805.00760
[20]: https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf
[21]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/td_lstm.py
[22]: https://github.com/wxue004cs/GCAE
[23]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/memnet.py
[24]: https://github.com/ThomasK427/DAuM-pytorch
[25]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/atae_lstm.py
[26]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/ian.py
[27]: https://github.com/hsqmlzno1/MGAN
[28]: https://arxiv.org/pdf/1902.09314.pdf
[29]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/aen.py
[30]: https://aclweb.org/anthology/D18-1380
[31]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/mgan.py