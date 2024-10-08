# Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers


## Setup

First, download and set up the repo:

```bash
git clone https://github.com/Juanerx/Q-DiT.git
cd Q-DiT
```
Then create the environment and install required packages:
```bash
conda create -n qdit python=3.8
conda activate qdit
pip install -r requirements.txt
pip install .
```


## Usage
If you want to use gptq or static quantization, calibration data should be generated by:
```bash
cd scripts
python collect_cali_data.py 
```

We can quantize the model:
```bash
bash quant_main.sh --image-size 256 --num-sampling-steps 50 --cfg-scale 1.5 --use_gptq
```


## BibTeX

```bibtex
@misc{chen2024QDiT,
      title={Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers}, 
      author={Lei Chen and Yuan Meng and Chen Tang and Xinzhu Ma and Jingyan Jiang and Xin Wang and Zhi Wang and Wenwu Zhu},
      year={2024},
      eprint={2406.17343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      url={https://arxiv.org/abs/2406.17343}, 
}
```


## Acknowledgments
This codebase borrows from [GPTQ](https://github.com/IST-DASLab/gptq.git), [Atom](https://github.com/efeslab/Atom) and [ADM](https://github.com/openai/guided-diffusion). Thanks to the authors for releasing their codebases!

