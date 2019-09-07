# Aspect-Based-Sentiment-Analysis

A paper list for aspect based sentiment analysis.

## Datasets

### SemEval-2014 Task 4

The [SemEval-2014 Task 4][45] contains two domain-specific datasets for laptops and restaurants, consisting of over 6K sentences with fine-grained aspect-level human annotations. [[data]][46]

The task consists of the following subtasks:

- Subtask 1: Aspect term extraction

- Subtask 2: Aspect term polarity

- Subtask 3: Aspect category detection

- Subtask 4: Aspect category polarity


### SentiHood

- **SentiHood**: Sentihood: Targeted aspect based sentiment analysis dataset for urban neighbourhoods. COLING 2016. [[paper]][43] [[data]][44]

### TOWE

- **TOWE**: Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling. NACCL 2019. [[paper]][55] [[data]][56]

### MAMS

- **MAMS**: A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis. EMNLP-IJCNLP 2019. [[data]][59]

## Paper List

### Aspect Base Sentiment Classification

#### RNN based model

- **TD_LSTM**: Effective LSTMs for Target-Dependent Sentiment Classification. COLING 2016. [[paper]][8] [[code]][21]

- **DMUE**: Recurrent Entity Networks with Delayed Memory Update for Targeted Aspect-based Sentiment Analysis. NAACL 2018. [[paper]][47] [[code]][48]

#### CNN based model

- **GCAE**: Aspect Based Sentiment Analysis with Gated Convolutional Networks. ACL 2018. [[paper]][2] [[code]][22]

#### Attention based model

- **MemNet**: Aspect Level Sentiment Classification with Deep Memory Network. EMNLP 2016. [[paper]][6] [[code]][23]

- **DAuM**: Enhanced Aspect Level Sentiment Classification with Auxiliary Memory. COLING 2018. [[paper]][11] [[code]][24]

- **AEN**: Attentional Encoder Network for Targeted Sentiment Classification. arXiv preprint 2019. [[paper]][28] [[code]][29]

- Progressive Self-Supervised Attention Learning forAspect-Level Sentiment Analysis. ACL 2019. [[paper]][64] [[code]][65]

#### RNN & Attention based model

- **ATAE_LSTM**: Attention-based LSTM for Aspect-level Sentiment Classification. EMNLP 2016. [[paper]][10] [[code]][25]

- **RAM**: Recurrent Attention Network on Memory for Aspect Sentiment Analysis. EMNLP 2017. [[paper]][41] [[code]][42]

- **IAN**: Interactive Attention Networks for Aspect-Level Sentiment Classification. IJCAI 2017. [[paper]][1] [[code]][26]

- **SysATT**: Effective Attention Modeling for Aspect-Level Sentiment Classification. COLING 2018. [[paper]][3]

- **MGAN**: Multi-grained Attention Network for Aspect-LevelSentiment Classification. ACL 2018. [[paper]][30] [[code]][31]

- **AOA_LSTM**: Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks. SBP-BRiMS 2018. [[paper]][32] [[code]][33]

- **Cabasc**: Content Attention Model for Aspect Based Sentiment Analysis. WWW 2018. [[paper]][36] [[code]][37]

#### RNN & CNN based model

- **TNet**: Transformation Networks for Target-Oriented Sentiment Classification. ACL 2018. [[paper]][34] [[code]][35]

#### RecursiveNN based model

- Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification. ACL 2014. [[paper]][4]

- PhraseRNN: Phrase Recursive Neural Network for Aspect-based Sentiment Analysis. EMNLP 2015. [[paper]][7]

#### Capsule network based model

- **CapsNet**: A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis. EMNLP-IJCNLP 2019. [paper] [[code]][59]

- Transfer Capsule Network for Aspect Level Sentiment Classification. ACL 2019. [[paper]][62] [[code]][63]

#### BERT base model

- **AEN_BERT / BERT_SPC**: Attentional Encoder Network for Targeted Sentiment Classification. arXiv preprint 2019. [[paper]][28] [[code]][29]

- **BERT_PT**: BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis. NAACL 2019. [[paper]][49] [[code]][50]

- **BERT-pair**: Utilizing BERT for Aspect-Based Sentiment Analysisvia Constructing Auxiliary Sentence. NAACL 2019. [[paper]][51] [[code]][52]

- **CapsNet-BERT**: A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis. EMNLP-IJCNLP 2019. [paper] [code][59]

#### Transfer learning model

- **MGAN**: Exploiting Coarse-to-Fine Task Transfer for Aspect-level Sentiment Classification. AAAI 2019. [[paper]][12] [[data]][27]

- **PRET & MULT**: Exploiting Document Knowledge for Aspect-level Sentiment Classification. ACL 2018. [[paper]][9] [[code]][38]

#### Multi-Task learning model

- An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis. ACL 2019. [[paper]][60] [[code]][61]

### Graph neural network based model

- **SDGCN**: Modeling Sentiment Dependencies with Graph Convolutional Networks for Aspect-levelSentiment Classification. arXiv preprint 2019. [[paper]][57] [[code]][58]

#### Hierarchical model

- Aspect Specific Sentiment Analysis using Hierarchical Deep Learning. NIPS 2014 workshop. [[paper]][5]

- A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis. EMNLP 2016. [[paper]][15]

#### Other

- Targeted Aspect-Based Sentiment Analysis via Embedding Commonsense Knowledge into an Attentive LSTM. AAAI 2018. [[paper]][13]

- Target-Sensitive Memory Networks for Aspect Sentiment Classification. ACL 2018. [[paper]][14]

- Modeling Inter-Aspect Dependencies for Aspect-Based Sentiment Analysis. NAACL 2018. [[paper]][16]

### Aspect Extraction

- **ABAE**: An Unsupervised Neural Attention Model for Aspect Extraction. ACL 2017. [[paper]][17] [[code]][39]

- **DECNN**: Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction. ACL 2018. [[paper]][18] [[code]][40]

- **Unified**: A Unified Model for Opinion Target Extraction and Target Sentiment Prediction. AAAI 2019. [[paper]][53] [[code]][54]

- Aspect Term Extraction with History Attention and Selective Transformation. IJCAI 2018. [[paper]][19]

- Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms. AAAI 2017. [[paper]][20]

- Annotation and Automatic Classification of Aspectual Categories. ACL 2019. [[paper]][66] [[code]][67]

- Exploring Sequence-to-Sequence Learning in Aspect Term Extraction. ACL 2019. [[paper]][68]

- DOER: Dual Cross-Shared RNN for Aspect Term-Polarity Co-Extraction. ACL 2019. [[paper]][69] [[code]][70]

### Target-oriented Opinion Words Extraction

- **IOG**: Target-oriented Opinion Words Extraction with Target-fused NeuralSequence Labeling. NAACL 2019. [[paper]][55] [[data]][56]

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
[32]: https://arxiv.org/pdf/1804.06536.pdf
[33]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/aoa.py
[34]: https://arxiv.org/pdf/1805.01086.pdf
[35]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/tnet_lf.py
[36]: http://delivery.acm.org/10.1145/3190000/3186001/p1023-liu.pdf?ip=210.75.253.235&id=3186001&acc=OPEN&key=33E289E220520BFB%2E6FFDCCEC948C43C2%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1555218179_72930af7a4eda051856a87b787ca5863
[37]: https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/cabasc.py
[38]: https://github.com/ruidan/Aspect-level-sentiment
[39]: https://github.com/ruidan/Unsupervised-Aspect-Extraction
[40]: https://github.com/yafangy/Review_aspect_extraction
[41]: https://www.aclweb.org/anthology/D17-1047
[42]: https://github.com/lpq29743/RAM
[43]: https://www.aclweb.org/anthology/C16-1146
[44]: https://github.com/uclmr/jack/tree/master/data/sentihood
[45]: http://alt.qcri.org/semeval2014/task4/
[46]: https://github.com/ThomasK427/aspect_extraction/tree/master/data/official_data/SemEval-2014
[47]: https://aclweb.org/anthology/N18-2045
[48]: https://github.com/liufly/delayed-memory-update-entnet
[49]: https://arxiv.org/pdf/1904.02232v1.pdf
[50]: https://github.com/howardhsu/BERT-for-RRC-ABSA
[51]: https://arxiv.org/pdf/1903.09588.pdf
[52]: https://github.com/HSLCY/ABSA-BERT-pair
[53]: https://arxiv.org/pdf/1811.05082.pdf
[54]: https://github.com/lixin4ever/E2E-TBSA
[55]: https://www.aclweb.org/anthology/N19-1259
[56]: https://github.com/NJUNLP/TOWE
[57]: https://arxiv.org/pdf/1906.04501.pdf
[58]: https://github.com/Pinlong-Zhao/SDGCN
[59]: https://github.com/siat-nlp/MAMS-for-ABSA
[60]: https://www.aclweb.org/anthology/P19-1048
[61]: https://github.com/ruidan/IMN-E2E-ABSA
[62]: https://www.aclweb.org/anthology/P19-1052
[63]: https://github.com/NLPWM-WHU/TransCap
[64]: https://www.aclweb.org/anthology/P19-1053
[65]: https://github.com/DeepLearnXMU/PSSAttention
[66]: https://www.aclweb.org/anthology/P19-1323
[67]: https://github.com/wroberts/annotator
[68]: https://www.aclweb.org/anthology/P19-1344
[69]: https://www.aclweb.org/anthology/P19-1056
[70]: https://github.com/ArrowLuo/DOER