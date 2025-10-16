<div align="center">
  
# Assembler: Scalable 3D Part Assembly via Anchor Point Diffusion

<a href="https://assembler3d.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
&ensp;
<a href="https://arxiv.org/abs/2506.17074"><img src="https://img.shields.io/badge/ArXiv-2506.17074-red"></a>
&ensp;
<a href="https://github.com/assembler3d/assembler3d.git"><img src="https://img.shields.io/badge/Code-Github-green"></a>
<br>

**[Wang Zhao<sup>1</sup>](https://thuzhaowang.github.io), [Yan-Pei Cao<sup>2</sup>](https://yanpei.me/), [Jiale Xu<sup>1</sup>](https://bluestyle97.github.io/),  [Yuejiang Dong<sup>1,3</sup>](https://scholar.google.com.hk/citations?user=0i7bPj8AAAAJ&hl=zh-CN), [Ying Shan<sup>1</sup>](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)**

<sup>1</sup>ARC Lab, Tencent PCG &ensp;&ensp;<sup>2</sup>VAST &ensp;&ensp;<sup>3</sup>Tsinghua University


**SIGGRAPH ASIA 2025**

</div>

---

## 🚩 Overview
This repository contains code release for our SIGGRAPH ASIA 2025 paper "Assembler: Scalable 3D Part Assembly via Anchor Point Diffusion".

<p align="center">
  <img src="assets/teaser.png" >
</p>

## ⚙️ Installation
We recommend using anaconda to install the dependencies:
```
conda create -n assembler python=3.10.16
conda activate assembler
conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 🚀 Usage
### Inference
To run the inference demo, simply use:
```
python ./scripts/demo.py --config ./configs/demo/demo.yaml --input_dir ./examples/4ef447cbb4a72f0a0e5941c9073c4baa0babd3f93ec55d62b040915f8bf3f49c --output_dir ./outputs/4ef447
```
This script runs for example data inside `./examples` from Toys4k dataset. You could use your own data to assemble. Put all the part meshes (in GLB format) and reference image (in PNG format) into a single folder, and set the `input_dir` argument to that folder.
