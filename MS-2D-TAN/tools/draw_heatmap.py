import numpy as np
import matplotlib.pyplot as plt
import os
from core.utils import iou

def load_illustration_map():
    # gt_time = [1.5, 2.0]
    gt_time = [1.5, 7.5]
    duration = 8
    time_unit = 1
    num_anchors = 8
    s_time = np.arange(0,duration,time_unit)[:,None].repeat(num_anchors, axis=1)
    duration = np.arange(1,num_anchors+1)[None,:].repeat(s_time.shape[0],axis=0)*time_unit
    pred_time = np.stack([s_time, s_time+duration], axis=2).reshape(-1,2).tolist()
    score_map = iou(pred_time, [gt_time]).reshape(s_time.shape[0], num_anchors)
    return score_map

def load_visualization_map():
    # score_map = np.load('results/example_a_gt_map.npy')[:82,:82]
    # score_map = np.load('results/example_a_pred_map.npy')[:82,:82]
    # score_map = np.load('results/example_b_gt_map.npy')[:30,:30]
    score_map = np.load('results/example_b_pred_map.npy')[:30,:30]
    return score_map

def drawing():
    # video_id = 'test-3951'
    # map_id = 'gate-4-1'
    # dataset_name = 'tacos'
    # score_map = np.load('results/{}/{}/{}.npy'.format(dataset_name, video_id, map_id))
    # figure_root = 'moment_localization/figures/{}/{}/'.format(dataset_name, video_id)


    score_map = load_illustration_map()
    length, num_anchors = score_map.shape
    for i in range(num_anchors):
        score_map[length-i-1,i+1:] = np.nan
    #

    fig, ax = plt.subplots(1, 1, figsize=(num_anchors, length))
    im = ax.imshow(score_map, cmap='Reds', vmin=0, vmax=1.0)#, aspect='auto'
    plt.grid(True)
    ax.set_xticks(np.arange(0,num_anchors,1)+0.5)
    ax.set_yticks(np.arange(0,length,1)+0.5)
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks_position('none')
    cbar = fig.colorbar(im, extend='max')
    cbar.cmap.set_under('#D4D4D4')
    plt.setp(ax.spines.values(), color='#C8C8C8')
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='#C8C8C8')
    plt.grid(color='#C8C8C8', linewidth=0.7)#, linestyle='-.'
    fig.tight_layout()
    # plt.show()
    fig.savefig(os.path.join('figures', '{}.png'.format("long_segments")), bbox_inches='tight')

if __name__ == '__main__':
    drawing()