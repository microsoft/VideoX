# SeqTrack Model Zoo

Here we provide the performance of the SeqTrack trackers on multiple tracking benchmarks and the corresponding raw results. 
The model weights and the corresponding training logs are also given by the links.

## Tracking
### Models

<table>
  <tr>
    <th>Model</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>LaSOT-ext<br>AUC (%)</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>VOT2020-bbox<br>EAO</th>
    <th>VOT2020-mask<br>EAO</th>
    <th>TNL2K<br>AUC (%)</th>
    <th>NFS<br>AUC (%)</th>
    <th>UAV<br>AUC (%)</th>
    <th>Models</th>
  </tr>
  <tr>
    <td>SeqTrack-B256</td>
    <td>69.9</td>
    <td>49.5</td>
    <td>74.7</td>
    <td>83.3</td>
    <td>29.1</td>
    <td>52.0</td>
    <td>54.9</td>
    <td>67.6</td>
    <td>69.2</td>
    <td><a href="https://drive.google.com/drive/folders/10PKs6aqSbVtb6aloYmtaN4aKMnmPF58P?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_kf2noaP3M9RHHCgG6XqBFejNkzBv2xB?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>SeqTrack-B384</td>
    <td>71.5</td>
    <td>50.5</td>
    <td>74.5</td>
    <td>83.9</td>
    <td>31.2</td>
    <td>52.2</td>
    <td>56.4</td>
    <td>66.7</td>
    <td>68.6</td>
    <td><a href="https://drive.google.com/drive/folders/10PKs6aqSbVtb6aloYmtaN4aKMnmPF58P?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_kf2noaP3M9RHHCgG6XqBFejNkzBv2xB?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>SeqTrack-L256</td>
    <td>72.1</td>
    <td>50.5</td>
    <td>74.5</td>
    <td>85.0</td>
    <td>31.3</td>
    <td>55.5</td>
    <td>56.9</td>
    <td>66.9</td>
    <td>69.7</td>
    <td><a href="https://drive.google.com/drive/folders/10PKs6aqSbVtb6aloYmtaN4aKMnmPF58P?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_kf2noaP3M9RHHCgG6XqBFejNkzBv2xB?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>SeqTrack-L384</td>
    <td>72.5</td>
    <td>50.7</td>
    <td>74.8</td>
    <td>85.5</td>
    <td>31.9</td>
    <td>56.1</td>
    <td>57.8</td>
    <td>66.2</td>
    <td>68.5</td>
    <td><a href="https://drive.google.com/drive/folders/10PKs6aqSbVtb6aloYmtaN4aKMnmPF58P?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/drive/folders/1_kf2noaP3M9RHHCgG6XqBFejNkzBv2xB?usp=sharing">log</a></td>
  </tr>
    


</table>

The downloaded checkpoints should be organized in the following structure
   ```
   ${SeqTrack_ROOT}
    -- checkpoints
        -- train
            -- seqtrack
                -- seqtrack_b256
                    SEQTRACK.pth.tar
                -- seqtrack_b256_got
                    SEQTRACK.pth.tar
                -- seqtrack_b384
                    SEQTRACK.pth.tar
                -- seqtrack_b384_got
                    SEQTRACK.pth.tar
                -- seqtrack_l256
                    SEQTRACK.pth.tar
                -- seqtrack_l256_got
                    SEQTRACK.pth.tar
                -- seqtrack_l384
                    SEQTRACK.pth.tar
                -- seqtrack_l384_got
                    SEQTRACK.pth.tar
   ```
### Raw Results
The [raw results](https://drive.google.com/drive/folders/15xrVifqG_idkXVxJOhUWq7nB5rzNxyO_?usp=sharing) are in the format [top_left_x, top_left_y, width, height]. Raw results of GOT-10K and TrackingNet can be 
directly submitted to the corresponding evaluation servers. The folder ```test/tracking_results/``` contains raw results and results should be organized in the following structure
   ```
   ${SeqTrack_ROOT}
    -- test
        -- tracking_results
            -- seqtrack
                -- seqtrack_b256
                    --lasot
                        airplane-1.txt
                        airplane-13.txt
                        ...
                    --lasot_extension_subset
                        atv-1.txt
                        atv-2.txt
                        ...
                -- seqtrack_b384
                    --lasot
                        airplane-1.txt
                        airplane-13.txt
                        ...
                ...
   ```
The raw results of VOT2020 should be organized in the following structure
   ```
   ${SeqTrack_ROOT}
    -- external
        -- vot20
            -- seqtrack
                -- results
                    --seqtrack_b256_ar
                    --seqtrack_b384_ar
                    ...
   ```
