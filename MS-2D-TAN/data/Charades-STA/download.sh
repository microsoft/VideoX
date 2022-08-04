wget http://ai2-website.s3.amazonaws.com/data/Charades_v1_features_rgb.tar.gz
tar -xvzf Charades_v1_features_rgb.tar.gz
python convert_vgg_features_to_hdf5.py
