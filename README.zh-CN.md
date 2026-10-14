# Trajectory Playground：同一目标，两条路线

[在线体验](https://rwlinno.github.io/trajectory-playground/) · [English](README.md)

一个精简的多模态生成教学项目：同一障碍物上、下两条路线都能绕行，但平方误差回归会把它们平均成穿过障碍物的直线。条件扩散则可以采样出不同路线。

![真实训练样本动画](docs/assets/preview.gif)

网页可切换 9 个预设场景、调整展示数量、重新抽样、回放轨迹，并显示几何有效率和有效路线覆盖。网页展示的是实际训练后的预生成样本，每个场景每个方法 64 条；不需要 GPU 或服务器，也不是实时模型推理。

## 这个项目有意保持简单

路径由直线加正弦弯曲组成，模型只学习一个弯曲幅度。端点由公式保证；障碍物为圆形，运动对象为点。它不是机械臂规划器，不验证关节、动力学或现实操作，不是先进方法排名。

同一份 8,192 条对称演示训练 MLP 和扩散模型。独立评估使用 128 个条件，每个条件 64 条样本。一次训练（seed 42）的结果：MLP 几何有效率 0%，扩散 98.33%，参考生成器 100%。失败样本全部保留。这个刻意构造的例子解释回归均值的问题，不能证明扩散普遍优于其他模型；简单混合模型也可以解决。

## 本地使用

仅看演示：`python3 -m http.server 8000 --directory docs`，打开 `http://localhost:8000`。

复现环境与训练：

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_artifacts.py
# 以下操作会覆盖已提交的模型和样本
bash scripts/run_demo_training.sh
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/render_preview.py
node --test tests/web.test.js
```

需要核对的指标文件：`artifacts/evaluation.json`（独立条件逐项评估与汇总）、`docs/assets/samples.json`（网页所有样本与场景指标）。`artifacts/models.pt` 为权重；校验脚本从权重重新生成全部样本和评估结果。环境和方法细节见英文 README。

MIT 协议。欢迎增加小而清楚的对比，例如混合密度网络、非对称模态权重和固定采样预算下的有效覆盖率。
