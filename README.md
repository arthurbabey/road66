# Project Road Segmentation

For this choice of project task, we provide a set of satellite images acquired 
from GoogleMaps. We also provide ground-truth images where each pixel is labeled 
as road or background. 

Your task is to train a classifier to segment roads in these images, i.e. 
assigns a label `road=1, background=0` to each pixel.

Submission system environment setup:

1. The dataset is available from the 
[CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).

2. Obtain the python notebook `segment_aerial_images.ipynb` from this github 
folder, to see example code on how to extract the images as well as 
corresponding labels of each pixel.

The notebook shows how to use `scikit learn` to generate features from each 
pixel, and finally train a linear classifier to predict whether each pixel is 
road or background. Or you can use your own code as well. Our example code here 
also provides helper functions to visualize the images, labels and predictions. 
In particular, the two functions `mask_to_submission.py` and 
`submission_to_mask.py` help you to convert from the submission format to a 
visualization, and vice versa.

3. As a more advanced approach, try `tf_aerial_images.py`, which demonstrates 
the use of a basic convolutional neural network in TensorFlow for the same 
prediction task.

Evaluation Metric:
 [F1 score](https://en.wikipedia.org/wiki/F1_score)
 
 
 ## Project structure
<pre>
.
├── README.md
├── checkpoints                             Trained models as .pt file
├── data            
│   └── train
│       ├── groundtruth                     Label images (400x400)
│       └── images                          Satelite images (400x400)
|   └── test
│       ├── predictions                     Predicted label images (608x608)
│       └── images                          Satelite images (608x608)
├── plots   
├── report                                  Final report
├── src                                     Source files (descriptions are in files directly)
│   ├── datasets                        
│   │   ├── aerial_dataset.py
│   │   └── patched_aerial_dataset.py
│   ├── helpers
│   │   ├── cross_val.py
│   │   ├── prediction.py
│   │   ├── search.py
│   │   ├── training.py
│   │   └── training_crossval.py
│   ├── kaggle
│   │   └── mask_to_submission.py
│   ├── models
│   │   └── rsm.py
│   │   └── uNet.py
│   ├── notebooks
│   │   ├── Pipeline.ipynb
│   │   ├── Pipeline_crossval.ipynb
│   │   ├── cnn.ipynb
│   │   ├── forest.ipynb
│   ├── postprocessing
│   │   ├── graph.py
│   │   ├── majority_voting.py
│   │   └── vectors.py
│   ├── preprocessing
│   │   ├── augmentation_config.py
│   │   ├── channels.py
│   │   ├── labeling.py
│   │   ├── loading.py
│   │   ├── patch.py
│   │   ├── prepare_images.py
│   │   └── rotation.py
│   ├── run.py                          Main file to train and generate predictions
│   └── visualization
│       └── helpers.py
└── submissions                         Generated CSV submissions for Kaggle
</pre>
