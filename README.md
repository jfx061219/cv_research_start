- Python：3.12.3
- Pytorch：GPU版本
- CUDA：支持GPU加速
- 虚拟环境：dl_env
```bash
conda env create -f environment.yml
conda activate dl_env
cv_research_start/
├── environment.yml    # conda环境导出文件
├── .gitignore         # git忽略文件配置
├── train.py           # 训练脚本
├── data/              # CIFAR10数据集（自动下载）
└── runs/              # Tensorboard日志
python train.py
