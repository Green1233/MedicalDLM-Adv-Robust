# MedicalDLM-Complexity-AdvRobust
This repository contains the <!--datasets and-->codes used in our study On the Role of Deep Learning Model Complexity in Adversarial Robustness of Medical Images.
## Usage
### 1. Check the requirements.
* python 3.7.6
* tensorflow-gpu 1.14.0
* keras 2.3.1
* numpy 1.18.1
* pandas 1.0.5
* scikit-learn 0.23.1
* scipy 1.5.2
* matplotlib 3.3.0
* pillow 7.2.0
* mlxtend 0.17.3
* cleverhans 2.1.0
### 2. Download the following datasets.
* [Chest X-ray](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* [Dermoscopy](https://www.kaggle.com/drscarlat/melanoma)
* [OCT](https://www.kaggle.com/paultimothymooney/kermany2018)
### 3. Train models.
python3 train_cbrLargeT.py<br/>
python3 train_resnet.py<br/>
<!--python3 train_resnet20.py<br/>
python3 train_resnet32.py<br/>
python3 train_resnet50.py<br/>-->
### 4. Adversarial attacks.
<!---# Attacks a pretrained DNN model with FGSM or PGD attack for a specified range of epsilon values<br/>
Generates saliency maps of test data for specified image index at each epsilon value<br/>
Generates decision boundaries of test data for specified image index at each epsilon value -->
python3 fgsm.py<br/>
python3 pgd.py
### Plot adversarial robustness curve
python3 plot_robustness.py





