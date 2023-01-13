# COVID-19-SSL
Source code of the paper "Data augmentation based semi-supervised method to improve COVID-19 CT classification"

# How to Run
1.  Prepare Datasets. My file tree is as follows:
ROOT:
├─datasets
│    ├─COVID-CT
│    │    ├─data
│    │    │    ├─COVID
│    │    │    │    covid_img1.png
│    │    │    │    covid_img2.png
│    │    │    │    covid_img3.png
│    │    │    │    ...
│    │    │    └─NonCOVID
│    │    │         Noncovid_img1.png
│    │    │         Noncovid_img2.png
│    │    │         Noncovid_img3.png
│    │    │         ...
│    │    ├─Data-split
│    │    │    ├─COVID
│    │    │    │    test.txt
│    │    │    │    train.txt
│    │    │    │    train_label.txt
│    │    │    │    train_unlabel.txt
│    │    │    └─NonCOVID
│    │    │         test.txt
│    │    │         train.txt
│    │    │         train_label.txt
│    │    │         train_unlabel.txt
│    │    └─label_unlabel_split.txt
│    ├─SARS-COV-2
│    │    ...
│    └─Harvard Dataverse
│         ...
├─model_result
├─covid_ct_main.py
├─data_process.py
├─misc.py
├─pseudo_labeling_util.py
└─train_util.py
The process file can be generated by the program, but the images need to be placed in the correct position.

2.  run covid_ct_main.py

# If this code is helpful to you, please cite this paper.
