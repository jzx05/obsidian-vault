---
title: WAM 评估体系图
date: 2026-06-04
tags:
  - WAM
  - 评估
  - 图示
---

# 📏 WAM 评估体系图

> WAM 的双重身份决定了它有**两套独立又交织**的评估维度：World Modeling（"想得对不对"）+ Action Policy（"做得好不好"）。

---

## 1. 两轴评估全景

```mermaid
flowchart LR
    classDef root fill:#fff2b3,stroke:#d4a017,color:#5a3c00,stroke-width:2px
    classDef wm  fill:#cfe8fc,stroke:#5b9bd5,color:#0b3d66
    classDef ac  fill:#d6f5d6,stroke:#3aa655,color:#1f5d2a
    classDef leaf fill:#fde4cf,stroke:#e8a87c,color:#5a3c00

    R(("WAM 评估")):::root

    R --> WM["🌐 World Modeling 能力"]:::wm
    R --> AC["🤖 Action Policy 能力"]:::ac

    %% World modeling
    WM --> WM1["视觉保真度<br/>Visual Fidelity"]:::leaf
    WM1 --> WM1a["PSNR · SSIM · LPIPS"]:::leaf
    WM1 --> WM1b["DreamSim · DINO · FVD"]:::leaf

    WM --> WM2["物理常识<br/>Physical Commonsense"]:::leaf
    WM2 --> WM2a["VideoPhy · PhyGenBench"]:::leaf
    WM2 --> WM2b["Physics-IQ · WorldModelBench"]:::leaf

    WM --> WM3["运动与轨迹<br/>Motion / Trajectory"]:::leaf
    WM3 --> WM3a["WorldScore · EWMBench"]:::leaf

    WM --> WM4["动作合理性<br/>Action Plausibility"]:::leaf
    WM4 --> WM4a["WorldVerse"]:::leaf
    WM4 --> WM4b["IDM Tuning Test"]:::leaf

    %% Action policy
    AC --> AC1["通用操作<br/>General Manipulation"]:::leaf
    AC1 --> AC1a["CALVIN · LIBERO"]:::leaf
    AC1 --> AC1b["Meta-World · RoboMimic"]:::leaf

    AC --> AC2["双臂 / 人形<br/>Bimanual & Humanoid"]:::leaf
    AC2 --> AC2a["RoboTwin · Bi-Play"]:::leaf

    AC --> AC3["移动操作<br/>Mobile Manip"]:::leaf
    AC3 --> AC3a["MobileALOHA · OK-Robot · ManipulaTHOR"]:::leaf

    AC --> AC4["接触 / 可形变<br/>Contact / Deformable"]:::leaf
    AC4 --> AC4a["GarmentLab · DefBench · DaXBench"]:::leaf

    AC --> AC5["真机评测<br/>Real-Robot Eval"]:::leaf
    AC5 --> AC5a["Real-X Suite"]:::leaf
```

---

## 2. 评估雷达图（示意 · 一目了然）

```mermaid
%%{init: {"theme": "base"}}%%
quadrantChart
    title WAM 评估目标 · 性能 vs 成本
    x-axis 低成本 --> 高成本
    y-axis 弱信号 --> 强信号
    quadrant-1 真机 + 大基准
    quadrant-2 真机小样本
    quadrant-3 仿真 + 视觉指标
    quadrant-4 大规模仿真基准
    PSNR/SSIM: [0.15, 0.20]
    FVD/LPIPS: [0.25, 0.40]
    PhyGenBench: [0.40, 0.60]
    WorldScore: [0.45, 0.65]
    CALVIN/LIBERO: [0.55, 0.75]
    WorldVerse: [0.50, 0.70]
    Real-Robot Suite: [0.90, 0.95]
    MobileALOHA: [0.80, 0.85]
```

---

## 3. 评估流程速查

```mermaid
flowchart TB
    classDef step fill:#cfe8fc,stroke:#5b9bd5,color:#0b3d66
    classDef out  fill:#d6f5d6,stroke:#3aa655,color:#1f5d2a
    classDef warn fill:#ffd6d6,stroke:#cc3333,color:#660000

    Start([拿到一个 WAM]):::step --> Q1{是否需要<br/>解耦评估?}:::step
    Q1 -- 否 --> EndOnly["仅看下游任务成功率"]:::warn
    Q1 -- 是 --> S1["① 视觉保真<br/>PSNR/SSIM/LPIPS/FVD"]:::step
    S1 --> S2["② 物理常识<br/>PhyGenBench"]:::step
    S2 --> S3["③ 运动一致<br/>WorldScore"]:::step
    S3 --> S4["④ 动作可达<br/>WorldVerse / IDM-Test"]:::step
    S4 --> S5["⑤ 仿真任务成功率<br/>CALVIN / LIBERO"]:::step
    S5 --> S6["⑥ 真机迁移<br/>Real-Robot"]:::step
    S6 --> Done([全面评估完成]):::out
```

---

## 4. 三个常见误区

| 误区 | 真相 |
|---|---|
| 视觉指标好 = 策略好 | ❌ FVD 高 ≠ 动作可达，需补 IDM-Test |
| 仿真成功率 = 真机成功率 | ❌ Sim2Real 差距常见，必做真机抽样 |
| 单一基准跑分 | ❌ 应交叉跑 fidelity + dynamics + policy |

---

**返回**：[[WAM综述概览.md]]
