# CMSS
## Getting Started

#### Installation

1. Download from GitHub

   ```bash
   git clone https://github.com/yy3020195524/CMSS.git
   
   cd CMSS
   ```

2. Create conda environment

   ```bash
   conda create --name CMSS python=3.8
   conda activate CMSS
   pip install -r requirements.txt
   # CUDA 11.8 
   conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   
   ```

## Datasets
>Note: You can download our datasets as follows, please download our pre-processing dataset of AMOS from [here](https://pan.quark.cn/s/c28cfc69b223) and put them into the folder 'dataset/dataset_amos/':
### 1. MS-CMRSeg 2019 dataset: [here](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html)
### 2. AMOS Dataset: [here](https://zenodo.org/records/7262581)

# Running Experiments
#### Pre-train
Our encoder and decoder use a Foundation model's [[link](https://github.com/ljwztc/CLIP-Driven-Universal-Model)] pre-trained weights [[link](https://pan.quark.cn/s/f88bfea5443a)]. Please download them and put them into the folder 'pretrained_model' before running the following script.



```bash
#### Training stage


## multi GPU for training with DDP 
python main.py --distributed

## single GPU for training 
python main.py

#### Testing stage
python test.py

```
