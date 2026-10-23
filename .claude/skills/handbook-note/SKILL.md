---
name: handbook-note
description: Create or reformat VLA-Handbook study notes in the Obsidian vault's Handbook_VLA folder using the vault's handbook template. Use when the user asks to write a handbook note, summarize a VLA paper/model/architecture in Obsidian, or clean up an existing Handbook_VLA note.
---

# Handbook VLA Note Skill

在 Obsidian vault 的 `Handbook_VLA/` 目录下创建或整理「手册式」学习笔记。目标是让每篇笔记都能在 30 秒内看懂核心、在 5 分钟内复述架构与关键数字。

## 模板位置

`templates/handbook.md`

该模板使用 **Templater** 语法：

- `title` 与 H1：`<% tp.file.title %>`
- `created`：`'<% tp.date.now("YYYY-MM-DD") %>'`

Templater 已配置为 `Handbook_VLA/` 的 folder template：在 Obsidian 里新建空白笔记时会自动套用。注意 `Trigger Templater on new file creation` 是 Obsidian 的**本机 localStorage 开关**，不能写进 vault 配置；首次使用需在 `设置 → Templater` 里手动打开（会弹一次安全确认）。若没有触发，打开命令面板执行 `Templater: Replace templates in the active file`。

直接用 Write 工具创建完整笔记时，不要保留 `<% %>` 占位符：把 `tp.file.title` 换成实际标题，把 `tp.date.now(...)` 换成实际日期。

## 工作流

1. **确定主题与文件名**
   - 目录：`Handbook_VLA/`
   - 命名：`<模型/主题> — <副标题>.md`，例如 `RT-1 — 开山之作.md`、`RT-2 — VLA 的诞生.md`
   - 保留用户已有的中文破折号 `—` 与空格风格；不要改成 `-` 或下划线。
   - 若用户是「整理现有笔记」，保留原文件名，不要重命名。

2. **读模板**
   - 先读 `templates/handbook.md`，按其中的 section 顺序填写。
   - 模板里的 `<% tp.file.title %>` 和 `<% tp.date.now(...) %>` 是 Templater 占位符；直接写文件时必须替换成实际值。
   - frontmatter 字段全部保留；`title` 与文件名（去掉 `.md`）一致。

3. **填 frontmatter**
   ```yaml
   ---
   title: "RT-1 — 开山之作"
   tags: [handbook, vla, robot-transformer]
   created: 2026-10-22
   source: "https://arxiv.org/abs/2212.06817"
   status: read
   related:
     - "[[RT-2 — VLA 的诞生]]"
   ---
   ```
   - `created` 用当天日期，格式 `YYYY-MM-DD`。
   - `status`: `reading`（初稿）/ `read`（已核对）/ `archived`（不再关注）。
   - `source` 优先 arXiv / DOI / 官方项目页；没有就留空，不要编造。
   - `related` 用 Obsidian wikilink；指向尚未创建的笔记也可以，后续会自然补上。

4. **写正文**
   - 标题：`# <与 title 相同>`。
   - **核心思想**：`> [!abstract]` callout，1-2 句，必须回答「解决什么问题 + 用什么方法 + 结果如何」。
   - **Eureka Moment**：`> [!tip]` callout，一句最反直觉、最值得记住的洞见。
   - **架构与信息流**：`## 🏗️ 架构与信息流`，用 ```` ```text ```` 代码块画 ASCII 图；从左到右或从上到下，标出输入、模块、输出、频率。不要用 Mermaid，除非用户明确要求。
   - **关键机制**：3-6 条 bullet，每条写「是什么 + 为什么这样设计」，不要只列名词。
   - **关键数据**：表格，至少覆盖参数量、训练数据、控制频率、主要结果；不确定就写「待补」，不要猜。
   - **公式补充**：`## 🧮 补充：...`，用 Obsidian MathJax：行间 `$$ ... $$`，行内 `$...$`；公式后必须解释符号和直觉。
   - **代码实现**：`## 🧑‍💻 代码实现`，必须包含：
     - 核心代码：用 `python` 代码块写最小实现骨架（可以是对齐官方实现的伪代码）；**每行代码尽量写行内注释**（中文/英文均可）
     - 官方实现对照表：模块 ↔ 官方文件/类 ↔ 说明
     - 自己动手 checklist：3-5 条可以实际执行的验证步骤
   - **局限与失败模式**：至少 2 条，写具体场景，不写「效果有待提升」这类空话。
   - **关联**：wikilink 到后续模型、相关概念、论文笔记；没有现成笔记就留 `[[ ]]` 占位。
   - **参考**：论文链接 + 项目页/代码；没有官方代码就写「论文未发布官方代码」。

5. **整理已有笔记时**
   - 保留原文所有事实、数字和引用，不要删改用户观点。
   - 只做结构、层级、callout、公式定界符、表格和链接的整理。
   - 原文的 `\[ ... \]` 改成 Obsidian 的 `$$ ... $$`；行内 `\( ... \)` 改成 `$...$`。
   - 原文的纯文本代码块改成 ```` ```text ````，避免 Obsidian 把 ASCII 图当正文折行。
   - 原文没有的内容不要为了「填满模板」而编造；用「待补」或删除可选 section。

## Obsidian 语法约定

- 内部链接用 `[[笔记名]]`，不要用相对路径 Markdown 链接。
- 外部链接用 `[文字](https://...)`。
- Callout：`> [!abstract]`、`> [!tip]`、`> [!warning]`，标题写在 callout 第一行。
- 高亮用 `**加粗**`；不要用 HTML。
- 公式块前后留空行，避免被当成普通文本。
- 表格用标准 Markdown 表格；列宽不要求对齐，但分隔行必须完整。

## 质量检查

完成前逐项确认：

- [ ] frontmatter 是合法 YAML，`title` 与文件名一致
- [ ] 有 `[!abstract]` 和 `[!tip]` 两个 callout
- [ ] 有 ASCII 架构图（`text` 代码块）
- [ ] 关键机制每条都解释了「为什么」
- [ ] 关键数据表没有编造数字；不确定处标「待补」
- [ ] 行间公式用 `$$`，行内公式用 `$`，且符号说明完整
- [ ] 有 `## 🧑‍💻 代码实现`，包含核心代码、官方实现对照表、自己动手 checklist
- [ ] 代码块标注了语言（`python` / `text`），且没有残留 `TODO`
- [ ] 局限至少 2 条，具体到场景
- [ ] 关联与参考 section 存在
- [ ] 没有把原始事实改错；改写前后数字一致

## 快速示例

用户说「写一篇 RT-2 的 handbook 笔记」时：

1. 读 `templates/handbook.md`
2. 创建 `Handbook_VLA/RT-2 — VLA 的诞生.md`
3. 填 frontmatter（source 用 `https://arxiv.org/abs/2307.15818`）
4. 按模板写：核心思想 → Eureka → 架构图 → 关键机制 → 关键数据 → 公式 → 局限 → 关联 → 参考
5. 运行上面的质量检查

## 模板骨架（模板缺失时重建）

````markdown
---
title: "<% tp.file.title %>"
tags: [handbook, vla]
created: '<% tp.date.now("YYYY-MM-DD") %>'
source:
status: reading
related: []
---

# <% tp.file.title %>

> [!abstract] 核心思想
> 一句话说清：它解决什么问题，用什么方法，结果如何。

> [!tip] Eureka Moment
> 最值得记住的反直觉洞见。

## 🏗️ 架构与信息流

```text
输入 → 模块 A → 模块 B → 输出
```

## 🔑 关键机制
- **机制 1**：是什么 + 为什么这样设计。

## 📊 关键数据

| 维度 | 数值 |
|---|---|
| 参数量 | 待补 |
| 训练数据 | 待补 |
| 控制频率 | 待补 |
| 主要结果 | 待补 |

## 🧮 补充：公式 / 推导（可选）

### 基本公式

$$
% TODO: 在这里写公式
$$

### 符号说明

| 符号 | 含义 |
|---|---|
|  |  |

### 直觉

## 🧑‍💻 代码实现

### 核心代码

```python
# 最小实现骨架：输入 → 模型 → 输出
```

### 官方实现对照

| 模块 | 官方文件 / 类 | 说明 |
|---|---|---|
|  |  |  |

### 自己动手

- [ ] 用随机张量跑通 forward，打印每一层 shape
- [ ] 对齐官方实现的关键细节
- [ ] 复现损失并 overfit 一个 batch

## 🧩 局限与失败模式
- 

## 🔗 关联
- 后续：[[ ]]
- 相关概念：[[ ]]

## 📚 参考
- 论文：
- 项目页：
````
