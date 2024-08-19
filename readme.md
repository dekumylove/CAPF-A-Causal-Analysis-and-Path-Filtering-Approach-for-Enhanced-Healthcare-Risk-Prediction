## CAPF: A Causal Analysis and Path Filtering Approach for Enhanced Healthcare Prediction

![version](https://img.shields.io/badge/version-v3.5-green)
![python](https://img.shields.io/badge/python-3.9.19-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.0.1-brightgreen)

This repository contains our implementation of **CAPF**.

### Data Format
The patient EHR data, **features** is formatted as 
```
[sample_num * tensor(1, feature_num)]
```
The **label** is formatted as
```
[sample_num * tensor(1, target_num)]
```
The graph data like **rel_index** is formatted as
```
tensor(sample_num, visit_num, feature_num, target_num, path_num, K, rel_num)
```
After obtaining the adjacent matrix, features, and labels of the MIMIC-III and MIMIC-IV, you can use datapreprocess_iii.py and datapreprocess_iv.py to generate the graph data.

### Benchmark Datasets

* [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
* [MIMIC-IV](https://physionet.org/content/mimiciv/3.0/)

### Baseline Models

| Model                | Code                                                                                              | Reference                                                                        |
|----------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| LSTM                 | model/lstm.py                                                                                     | [LSTM]()                                                                         |
| Dipole               | model/Dipole.py                                                                                   | [Dipole](https://arxiv.org/pdf/1706.05764)                                       |
| Retain               | model/retain.py                                                                                   | [Retain](https://arxiv.org/pdf/1608.05745)                                       |
| MedPath              | model/medpath.py                                                                                  | [MedPath](https://dl.acm.org/doi/pdf/10.1145/3442381.3449860)                    |
| GraphCare            | model/graphcare.py                                                                                | [GraphCare](https://arxiv.org/pdf/2305.12788)                                    |
| HAR                  | model/stageaware.py                                                                               | [HAR](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10236511)         |

#### Running Scripts

The running scripts are available in run.sh. 

##### Best AUC and F1 for CAPF

The running scripts are available in "Section 1: Best AUC and F1 for CAPF" of run.sh.
Be reminded to save the checkpoints in the format of ".ckpt" after the model is trained.

#####  Best AUC and F1 for Baseline Methods

The running scripts are available in "Section 2: Best AUC and F1 for Baseline Methods" of run.sh.

##### Interpretation

The running scripts are available in "Section 3: Interpretation" of run.sh.
The significant paths and the corresponding attention values will be printed out during the evaluate stage.

#### Requirement

* python>=3.9.19
* PyTorch>=2.0.1