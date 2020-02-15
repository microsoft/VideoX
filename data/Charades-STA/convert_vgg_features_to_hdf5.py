import h5py
import os
from tqdm import tqdm
import glob
import numpy as np

if __name__ == '__main__':
    data_root = "."
    h5_file = h5py.File(os.path.join(data_root, 'charades_vgg_rgb.hdf5'), 'w')
    feat_path = os.path.join(data_root, 'Charades_v1_features_rgb/*')

    with open(os.path.join(data_root, 'valid_videos.txt')) as f:
        valid_vids = [line.split('\n')[0] for line in f.readlines()]

    pbar = tqdm(total=len(valid_vids))
    for vid_path in glob.glob(feat_path):
        vid = vid_path.split('/')[-1]
        if vid not in valid_vids:
            continue
        feats = []
        for txt_path in sorted(glob.glob(os.path.join(vid_path,'*'))):
            with open(txt_path) as txt_file:
                line = txt_file.readline()
                feat = [float(i) for i in line.split(' ')]
                feats.append(feat)
        feats = np.array(feats)

        h5_file.create_dataset(vid, data=feats, compression="gzip")
        pbar.update(1)
    pbar.close()
    h5_file.close()
