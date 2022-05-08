# cs598_f22_project


### Citation to the original paper

Xueping Peng, Guodong Long, Tao Shen, Sen Wang, Jing Jiang, and Chengqi Zhang. 2020. Bitenet: Bidirectional temporal encoder network to predict medical outcomes. CoRR, abs/2009.13252.

### Link to the original paper’s repo 

https://github.com/Xueping/BiteNet

### Dependencies

Python 3.8+
PyTorch v1.1.0 or later

### Data download instruction

The training data (MIMIC-III) used in the project is publically available with retrictions at https://physionet.org/content/mimiciii/1.4/.

### Preprocessing code

Preprocessing code and libraries are in `data_processing` folder and can be retrieved using `data_processing.load_data` command.

### Training & Evalue code 

All baseline and target model codes, including data loader, models, and training and evaluation code are executable in each Jupyter notebook.

### Table of results

*Result Part 1* 

| Model    | Diagnosis Precision      | 
|----------|:-------------:|
| RNN      |  0.628        | 
| BRNN     |  0.663        | 
| RETAIN   |  0.536        | 
| Deepr    |  0.659        | 
| SAnD     |  0.612        | 
| BiteNet  |  0.588        | 

*Result Part 2* 

| Model    | Diagnosis Precision      | 
|----------|:-------------:|
| RNN      |  0.203        | 
| BRNN     |  0.220        | 
| RETAIN   |  0.333        | 
| Deepr    |  0.358        | 
| SAnD     |  0.304        | 
| BiteNet  |  0.220        | 

### Citation to the baseline model codes

SAnD: https://github.com/khirotaka/SAnD 

RNN and BRNN: https://www.coursera.org/learn/cs598-deep-learning-for-healthcare/programming/iTiOz/homework-3-recurrent-neural-network
