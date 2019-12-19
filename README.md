# Project Road Segmentation

For this project task, we had to train a classifier to segment roads in satellite images from GoogleMaps, i.e assigns a label `road=1, background=0` to each pixel. provide a set of satellite images acquired 


The dataset is available from the 
[CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).


Evaluation Metric:
 [F1 score](https://en.wikipedia.org/wiki/F1_score)
 
 
 # How to reproduce our best submission : 
 
 First our best model was too heavy to be uploaded on github you can downloaded from `Google Drive` using this link : https://drive.google.com/open?id=1-800K4wciXJA47NDFPp-DHhxDoi8CGSb then you have to put it in the provided `best_model` folder.
 
Then you'll be able to run the `run.py` which allows you to either obtain prediction using our pretrained model or running the model from scratch by running with the `-train` flag.

You can also get the similar results by running the `notebook/unet.ipynb` on Google Colab. 
 
 # Contributors

- Benoît Hohl [@bentoCH](https://github.com/bentoCH)
- Joël Daout [@joeldaout](https://github.com/joeldaout)
- Arthur Babey [@arthurbabey](https://github.com/arthurbabey)

 # Python and external libraries

This project has been developped using `Python 3.6`.
We use the following libraries with the mentionned version.

`Tensorflow : '2.1.0-rc1', 
 Sklearn : '0.21.3', 
 Numpy : '1.17.4', 
 Scikit-image : '0.15.0'
 Keras : '2.3.1')`


 
 # Project structure
<pre>
.
├── README.md
│                           
├── datasets            
│   └── training
│   │    ├── groundtruth                     Label images (400x400)
│   │    └── images                          Satelite images (400x400)
|   └── test_set_images                      Satelite images (608x608)
│                               
│  
├── report                                  Final report
├── scripts                                   
│   │──helpers_unet.py
│   │  
│   │──│unet.py 
│   │ 
│   ├── notebooks
│   │   ├── unet.ipynb  
│   │
│   └── best_model                      Folder to place our best model download from the link mentionned above
│         
└── submissions                         Best submission on AIcrowd
</pre>
