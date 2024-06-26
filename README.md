# Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers


## Setup

First, download and set up the repo:

```bash
git clone https://github.com/Juanerx/Q-DiT.git
cd Q-DiT
```

## Usage
```bash
cd scripts
bash quant_main.sh --image-size 256 --num-sampling-steps 50 --cfg-scale 0
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

