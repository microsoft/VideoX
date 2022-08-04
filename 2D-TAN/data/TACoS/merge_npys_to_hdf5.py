import glob
import h5py
import numpy as np
import os
import tqdm
import json

def convert_tall_c3d_features(sampling_rate):
    stride = sampling_rate//5
    data_root = "./data/TACoS/"
    hdf5_file = h5py.File(os.path.join(data_root, 'tall_c3d_{}_features.hdf5'.format(sampling_rate)), 'w')
    with open(os.path.join(data_root,'train.json')) as json_file:
        annotation = json.load(json_file)
    with open(os.path.join(data_root, 'val.json')) as json_file:
        annotation.update(json.load(json_file))
    with open(os.path.join(data_root, 'test.json')) as json_file:
        annotation.update(json.load(json_file))
    pbar = tqdm.tqdm(total=len(annotation))
    for vid, anno in annotation.items():
        video_feature = []
        for i in range(0,(anno['num_frames']-sampling_rate)//stride+1):
            s_idx = i*stride+1
            e_idx = s_idx + sampling_rate
            clip_path = os.path.join(data_root, 'Interval64_128_256_512_overlap0.8_c3d_fc6','{}_{}_{}.npy'.format(vid, s_idx, e_idx))
            frame_feat = np.load(clip_path)
            video_feature.append(frame_feat)
        video_feature = np.stack(video_feature)
        hdf5_file.create_dataset(vid, data=video_feature, compression="gzip")
        pbar.update(1)
    pbar.close()
    hdf5_file.close()
if __name__ == '__main__':
    convert_tall_c3d_features(64)
