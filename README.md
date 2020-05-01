# COMP47650 Deep Learning Project: Audio Tagging Task

This project is about tagging audio clips that are classified with one of 41 classes. The project 
implements a few classifiers. We present audio classifiers based on CNNs and VGGish architecture.
Each network can take log-mel, mfcc, or chroma features as input. Predictions (over 41 classes) provided 
by classifiers are considered as their output. 

## Getting Started

Please follow these instructions to install all requirements and use the software correctly.

### Requirements

The project is written in Python 3.8.1 programming language. Neural networks are implemented using 
[PyTorch](https://pytorch.org/) framework. [LibROSA](https://librosa.github.io/librosa/#librosa) 
package for music and audio analysis is used to extract the features. We also make use 
[scikit-learn](https://scikit-learn.org/stable/) package to compute evaluation measures of proposed 
classifiers.

Datasets must be used from [Zenodo](https://zenodo.org/record/2552860#.XqxEMjMo-uV) and placed into
folders as described in ```hparams.yaml```. Namely, ```Datasets``` must include directory ```FSDKaggle2018.audio_train``` with training audios, ```FSDKaggle2018.audio_test``` with test audios,
```FSDKaggle2018.meta``` with csv meta-files ```train_post_competition.csv``` and ```test_post_competition_scoring_clips.csv```. We highly reccoment to keep this structure.

### Installing

To install all required modules of the appropriate version, please run the following command:

```
pip install -r requirements.txt
```

## Running

```bash
python main.py train --model={VGGish,CNN} --features={log_mel,mfcc,chroma} --validate --manually_verified_only --shuffle --cuda --verbose
```

```bash
python main.py validation --model={VGGish,CNN} --features={log_mel,mfcc,chroma} --epoch=EPOCH --validated --manually_verified_only --shuffle --cuda --verbose
```

```bash
python main.py test --model={VGGish,CNN} --features={log_mel,mfcc,chroma} --epoch=EPOCH --validated --manually_verified_only --cuda --verbose
```

## Plotting the results

```bash
python plot.py train --model={VGGish,CNN} --features={log_mel,mfcc,chroma} --validated --manually_verified_only --latex --verbose
```

```bash
python plot.py validation --model={VGGish,CNN} --features={log_mel,mfcc,chroma} --epoch=EPOCH --validated --manually_verified_only --latex --verbose
```

```bash
python plot.py test --model={VGGish,CNN} --features={log_mel,mfcc,chroma} --epoch=EPOCH --validated --manually_verified_only --latex --verbose
```

Use ```python {main,plot}.py {train,validation,test} -h[--help]``` for detail.
 
## Built With

* [PyTorch](https://pytorch.org/)
* [LibROSA](https://librosa.github.io/librosa/#librosa)
* [scikit-learn](https://scikit-learn.org/stable/)


## Authors

* **Roman Overko**
* **Dmytro Mishagli**
* **Xuesong Zhang**

