# COMP47650 Deep Learning Project: Audio Tagging Task

This project is about tagging audio clips that are classified with one of 41 classes. The project 
implements a few classifiers. We present audio classifiers based on CNN, ResNet, and VGGish architecture.
Each network can take raw, log-mel or mfcc features as input. Predictions (over 41 classes) provided 
by classifiers are considered as their output. 

## Getting Started

Please follow these instructions to install all requirements and use the software correctly.

### Requirements

The project is written in Python 3.8.1 programming language. Neural networks are implemented using 
[PyTorch](https://pytorch.org/) framework. [LibROSA](https://librosa.github.io/librosa/#librosa) 
package for music and audio analysis is used to extract log-mel and mfcc features. We also make use 
[scikit-learn](https://scikit-learn.org/stable/) package to compute evaluation measures of proposed 
classifiers.

### Installing

To install all required modules of the appropriate version, please run the following command:

```
pip install -r requirements.txt
```
Say where to put the datasets

Say about directory structure

```bash
b
```

No need to do additional steps of installation, the scripts are ready to run.

## Running the tests

```bash
python main.py train --model=VGGish --features=log_mel --validate --manually_verified_only --shuffle 
--cuda --verbose
```

```bash
python main.py validation --model=VGGish --features=log_mel --epoch=10 --validated 
--manually_verified_only --cuda --verbose
```

```bash
python main.py test --model=VGGish --features=log_mel --epoch=10 --validated 
--manually_verified_only --cuda --verbose
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

