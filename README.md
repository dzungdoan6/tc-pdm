# About
This is the Pytorch implementation of the paper **TC-PDM: Temporally Consistent Patch Diffusion Models for Infrared-to-Visible Video Translation**
# Installation
* Clone this repo, suppose the source code is saved in `[root]/tc-pdm`
* Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment
    ```
    conda create --name tc-pdm python=3.9
    conda activate tc-pdm
    ```
* Install pytorch
    ```
    pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -f https://download.pytorch.org/whl/cu118/torch_stable.html
    ```
* Install remaining packages
    ```
    pip install -r requirements.txt
    ```

# Dataset preparation
- Download the KAIST dataset [here](https://soonminhwang.github.io/rgbt-ped-detection/)
- Prepare data in the following format. 
```
    ├── data
    |    └── KAIST
    |         ├── set00            # all images of set00
    |         |    ├── lwir            # infrared images
    |         |    └── visible         # visible images           
    |         |
    |         ├── set00.txt        # already included in the repo
    |         |
    |         ├── set06_V002       # all images of V002 in set06
    |         |    ├── lwir            # infrared images
    |         |    └── visible         # visible images         
    |         |
    |         ├── set06_V002.txt   # already included in the repo
    
```
Remember to resize and center-crop all images to 256x256; see function `resize_to_320x256_crop_to_256x256(image_path)` in `utils/data_utils.py`

# Data preprocessing
### Extract segmentation logits

* Clone [dinov2](https://github.com/facebookresearch/dinov2). Suppose it is saved in `[root]/dinov2`
* Follow its instruction to install conda env `dinov2-extras`. We need `extras` dependency for semantic segmentation
* Copy `extract_seg_logits.py` from `[root]/tc-pdm` to `[root]/dinov2`
* Extract seg. logits on training and testing data
    ```
    cd [root]/dinov2
    conda deactivate                    # deactivate current env
    conda activate dinov2-extras        # activate dinov2 env
    python3 extract_seg_logits.py --base-dir [root]/tc-pdm/data/KAIST/ --set-name set00
    python3 extract_seg_logits.py --base-dir [root]/tc-pdm/data/KAIST/ --set-name set06_V002
    ```
* Once it is finished, we could find folder `lwir_seg` in `[root]/tc-pdm/data/set00` and `[root]/tc-pdm/data/set06_V002`, storing seg. logits. Note: please reserve around **750 GB** for this.

### Extract dense correspondences
* Clone [VideoFlow](https://github.com/XiaoyuShi97/VideoFlow). Suppose it is saved in `[root]/videoflow`
* Follow its instruction to install its conda env. Suppose it is named `videoflow`.
* Download the pretrained model. In the paper, we tried the model trained on Sintel
* Copy `extract_flow.py` from `[root]/tc-pdm` to `[root]/videoflow`
* Estimate optical flow on testing video
    ```
    cd [root]/videoflow
    conda deactivate                # deactivate current env
    conda activate videoflow        # activate videoflow env
    python3 extract_flow.py --base-dir [root]/tc-pdm/data/KAIST --set-name set06_V002
    ```
* Once it is finished, we could find the folder `lwir_flow` in `[root]/tc-pdm/data/set06_V002`
* Establish dense correspondences from optical flow
    ```
    cd [root]/tc-pdm
    conda deactivate                # deactivate current env
    conda activate tc-pdm           # activate tc-pdm
    python3 flows2matches.py --base-dir [root]/tc-pdm/data/KAIST --set-name set06_V002
    ```
* Once it is finished, we could find the folder `lwir_matches` and `lwir_inliers` in `[root]/tc-pdm/data/set06_V002`

# Usage
### Training
```
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=1 train_diffusion.py --config kaist.yml --txt set00.txt
```

Models will be saved in `[root]/tc-pdm/work_dir/KAIST/weights/`

### Testing
```
python3 eval_diffusion.py --config kaist.yml --resume /path/to/model --txt set06_V002.txt --resdir /path/to/results --tb
```

# Acknowledgments
This code is based on [DPIR](https://github.com/IGITUGraz/WeatherDiffusion) and [DDIM](https://github.com/ermongroup/ddim). A huge thanks to them!
