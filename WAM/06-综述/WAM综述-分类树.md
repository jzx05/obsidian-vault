---
title: WAM 分类树（Taxonomy）
date: 2026-06-04
tags:
  - WAM
  - 分类
  - 图示
---

# 🌳 WAM 分类树（Taxonomy）

> 综述 Fig.2 的精简可视化版：六大维度 × 代表方法。

---

## 1. 全景分类树

```mermaid
flowchart LR
    classDef root fill:#fff2b3,stroke:#d4a017,color:#5a3c00,stroke-width:2px
    classDef l1   fill:#cfe8fc,stroke:#5b9bd5,color:#0b3d66,stroke-width:1px
    classDef l2   fill:#d6f5d6,stroke:#3aa655,color:#1f5d2a
    classDef l3   fill:#fde4cf,stroke:#e8a87c,color:#5a3c00

    R(("WAM<br/>World Action Models")):::root

    R --> BG["📜 Background"]:::l1
    R --> AR["🏗️ Architecture"]:::l1
    R --> TR["📚 Training Data"]:::l1
    R --> EV["📏 Evaluation"]:::l1
    R --> OC["🔮 Open Challenges"]:::l1

    %% Background
    BG --> BG1["VLA Foundations"]:::l2
    BG --> BG2["World Model Foundations"]:::l2
    BG2 --> BG2a["Action-conditioned WM"]:::l3
    BG2 --> BG2b["Language-conditioned WM"]:::l3
    BG2 --> BG2c["Embodied WM (sim)"]:::l3

    %% Architecture
    AR --> AR1["🔵 Cascaded WAM"]:::l2
    AR1 --> AR1a["Pixel Prediction<br/>UniPi · SuSIE · UniVLA"]:::l3
    AR1 --> AR1b["Geometric / Flow<br/>AVDC · Im2Flow2Act · 3DFlowAction"]:::l3
    AR1 --> AR1c["Latent State<br/>VPP · SVAM · CoTraining"]:::l3

    AR --> AR2["🟢 Joint WAM"]:::l2
    AR2 --> AR2a["Autoregressive"]:::l3
    AR2a --> AR2a1["Decoupled Repr."]:::l3
    AR2a --> AR2a2["Unified Discrete"]:::l3
    AR2a --> AR2a3["Latent Repr."]:::l3
    AR2 --> AR2b["Diffusion"]:::l3
    AR2b --> AR2b1["Unified Stream<br/>UVA · PAD"]:::l3
    AR2b --> AR2b2["Cross-Attention<br/>DreamGen"]:::l3
    AR2b --> AR2b3["Hidden Coupled<br/>DyWA · AdaWorld"]:::l3
    AR2b --> AR2b4["Shared Repr.<br/>MotoDiff"]:::l3

    %% Training data
    TR --> TR1["Robot Teleop<br/>Open-X · Bridge"]:::l2
    TR --> TR2["Portable UMI<br/>UMI · DexUMI · FastUMI"]:::l2
    TR --> TR3["Simulation<br/>RoboCasa · ManiSkill"]:::l2
    TR --> TR4["Human / Ego-Centric<br/>Ego4D · EpicKitchens · RH20T"]:::l2

    %% Evaluation
    EV --> EV1["World Modeling"]:::l2
    EV1 --> EV1a["Visual Fidelity<br/>PSNR · SSIM · LPIPS · FVD"]:::l3
    EV1 --> EV1b["Object Dynamics<br/>VideoPhy · PhyGenBench"]:::l3
    EV1 --> EV1c["Motion / Trajectory<br/>WorldScore · EWMBench"]:::l3
    EV1 --> EV1d["Action Plausibility<br/>WorldVerse · IDM-Test"]:::l3

    EV --> EV2["Action Policy"]:::l2
    EV2 --> EV2a["Single-Arm<br/>CALVIN · LIBERO"]:::l3
    EV2 --> EV2b["Bimanual / Humanoid<br/>RoboTwin · BiPlay"]:::l3
    EV2 --> EV2c["Mobile Manip<br/>MobileALOHA · OK-Robot"]:::l3
    EV2 --> EV2d["Contact / Deformable"]:::l3
    EV2 --> EV2e["Real-Robot Eval"]:::l3

    %% Open Challenges
    OC --> OC1["架构 & 表征耦合"]:::l3
    OC --> OC2["多模态状态"]:::l3
    OC --> OC3["数据-课程-混合"]:::l3
    OC --> OC4["长程层次抽象"]:::l3
    OC --> OC5["推理效率"]:::l3
    OC --> OC6["评估方法学"]:::l3
    OC --> OC7["安全部署"]:::l3
```

---

## 2. Joint WAM 内部 4 种耦合方式（Diffusion 派）

```mermaid
flowchart TB
    classDef obs fill:#eef2ff,stroke:#6c7bd6,color:#1a237e
    classDef vid fill:#cfe8fc,stroke:#5b9bd5,color:#0b3d66
    classDef act fill:#d6f5d6,stroke:#3aa655,color:#1f5d2a
    classDef hid fill:#fff2b3,stroke:#d4a017,color:#5a3c00

    subgraph S1[" ① Unified Stream "]
        U1["DiT 单流<br/>UVA · PAD"]:::hid
        U1 --> V1[("Video Tokens")]:::vid
        U1 --> A1[("Action Tokens")]:::act
    end

    subgraph S2[" ② Cross-Attention Coupled "]
        V2[("Video DiT")]:::vid <--> A2[("Action DiT")]:::act
    end

    subgraph S3[" ③ Hidden-State Coupled "]
        V3[("Video Branch")]:::vid -- 隐状态注入 --> A3[("Action Branch")]:::act
    end

    subgraph S4[" ④ Shared Representation "]
        SR[("共享表征 z")]:::hid
        SR --> V4[("Video Decoder")]:::vid
        SR --> A4[("Action Decoder")]:::act
    end
```

---

## 3. 选型速查表

| 你的需求 | 推荐路线 | 代表工作 |
|---|---|---|
| 解释性强、可视化未来 | Cascaded · Pixel | UniPi, SuSIE |
| 几何/接触敏感 | Cascaded · Flow | AVDC, Im2Flow2Act |
| 训练-推理一致、低延迟 | Joint · AR | GR-1, UVA |
| 高保真、多模态 | Joint · Diffusion | DreamGen, DyWA |
| 大规模人类视频预训练 | Cascaded · Latent | VPP, OneVLA |

---

**返回**：[[WAM综述概览.md]]
