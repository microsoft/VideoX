# Alpha-Refine
## Introduction
Alpha-Refine is the winner of the VOT Real-Time Challenge 2020, which has great ability to predict high-quality masks. 
Following STARK, we combine the SeqTrack tracker with Alpha-Refine to test on the VOT2020 benchamark.

## Installation
After the environment has been installed according to the README.md of SeqTrack, you only need to install a few more packages as shown below.

* Install ninja-build for Precise ROI pooling  
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.

* Install the Precise ROI pooling
```
cd ltr/external
git clone https://github.com/vacancy/PreciseRoIPooling.git
cd ../..
```
* Add the project path to environment variables
```
export PYTHONPATH=<absolute_path_of_AR>:$PYTHONPATH
```

* Setup the environment  

Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  

* Download the pre-trained Alpha-Refine network  
Download the network for [Alpha-Refine](https://drive.google.com/open?id=1qOQRfaRMbQ2nmgX1NFjoQHfXOAn609QM) 
and put it under the ltr/checkpoints/ltr/ARcm_seg/ARcm_coco_seg_only_mask_384 dir.

