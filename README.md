Python version used: Python 3.12.2
Libraries needed: argparse, matplotlib, numpy, scipy

The 'task1.py' program handles 2 different experiment types:
1. Generation and display of a MFCC based on 1 audio file.
2. Evaluating MFCCs as basis for speaker identification alongside a kNN classifier (printing individual F1 scores and a macro-F1 score).

The following arguments are required for the program to run:
- ```--experiment```: Provide 'experiment1' for experiment type 1, provide 'experiment2' for experiment type 2.
- ```--parameter_configuration```: Provide a value from the list below corresponding to the desired parameters for the MFCC calculator.

It is expected that this code is to be executed from within the task1/code directory in order to ensure the relative filepaths for the speaker data is accessible.

An example of how to run this code is as follows:
```python task1.py --experiment experiment1 --parameter_configuration 0```