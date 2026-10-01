# 机器学习实现快照

本目录提供项目内部的观察/动作编码器、完整动作模型、V73 actor attention 扩展，以及一个可独立运行的 CPU 合成输入示例。三个实现文件保留原 basename 和原字节；SHA256 见 [机器可读实验记录](../data/ml_experiment.json)。实现过程使用 AI 编程工具辅助，本展示不声称全部代码由本人独立手写。

## 运行

需要 Python、NumPy 和 PyTorch。已有环境中直接运行：

```bash
python ml/smoke.py
```

新环境的依赖声明位于 `ml/requirements.txt`，可使用 `python -m pip install -r ml/requirements.txt` 安装。示例强制 CPU，随机初始化 hidden=32 的小模型；依次演示动作 codec 往返、teacher-forced forward、masked loss、一次反向传播和自由市场序列解码，没有 optimizer 更新。

该示例只验证张量接口和梯度流程，不复现历史训练、不评估比赛收益。已在 CPU 上运行通过，回执见 [smoke_result.json](smoke_result.json)，没有安装依赖、调用 GPU 或训练历史模型。不同依赖版本的数值输出可能略有差异。历史实验使用 hidden=192，相关数据规模、训练配置、验证指标和闭环失败结果另见实验记录与报告。

## 代码关系与复现范围

- `replay_learning/full_action_schema.py`：仅 NumPy/标准库，8,331 维全局观察、32×56 actor 特征、动作词表及编解码。
- `replay_learning/full_action_model.py`：全局 MLP/GRU、单位动作与 hands 长度输出、自回归市场 decoder、masked 交叉熵。
- `replay_learning/full_action_model_v73.py`：在活跃 actor 间使用一层四头 masked self-attention。

快照不包含教师策略、历史回放、训练权重或比赛 Agent。历史数据 builder 后来已修改，其当前源码 SHA 与当时 manifest 不同；本目录不承诺从零一键重现原训练数据或 88.25% 指标。历史验证集属于开发集，没有 sealed test；最终比赛 Agent 也不是这个 BC 神经网络。
