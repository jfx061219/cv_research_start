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
# 实验日志 exp001
## Config（超参数）
- 数据集：CIFAR-10
- 模型：SimpleCNN
- 随机种子：42
- Batch size：128
- 优化器：SGD，lr=0.1，momentum=0.9
- 最大训练轮数：100
## 实验截图
![TensorBoard曲线](https://imgchr.com/i/pnU4utS)
![TensorBoard曲线](https://imgchr.com/i/pnU4Kfg)
![混淆矩阵](https://imgchr.com/i/pnU4QpQ)
使用 SimpleCNN 在 CIFAR‑10 数据集进行训练，模型最优验证集准确率达到“86.9%”。训练损失持续下降，验证准确率稳步提升，后期验证指标小幅震荡但整体稳定，没有出现明显过拟合现象，模型收敛效果良好。

