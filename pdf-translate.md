---
name: pdf-translate
description: 将 PDF 文件翻译为另一种语言，保留原始排版、矢量图形、表格和布局。自动检查并安装所需环境。
---

# PDF 翻译技能

当用户输入 `/pdf-translate` 时，执行以下完整流程。

## 第一步：确认参数

向用户确认以下信息（如果用户已在消息中提供则跳过）：

1. **PDF 文件路径**（必填）— 用户可能直接拖拽文件或给出路径
2. **源语言**（默认 `en` 英文）
3. **目标语言**（默认 `zh` 中文）
4. **翻译服务**（默认 `google`，免费无需密钥）

如果用户没有提供 PDF 路径，主动询问。

## 第二步：环境检查与自动安装

**按顺序检查，缺少什么装什么，不要跳过。**

### 2.1 检查 Python（需要 3.10+）

依次尝试以下方式，找到一个能用的就记下来：

```bash
# 优先检查系统 PATH 中的 python
python --version
python3 --version

# 备选：py launcher (Windows)
py -3.12 --version
py -3.11 --version
py -3.10 --version
```

如果找到的 Python 版本 >= 3.10，记下完整路径，跳到 2.2。

如果找不到任何 Python 3.10+，**告诉用户需要安装 Python 3.10+**，并给出安装指引：
- Windows: https://www.python.org/downloads/
- macOS: `brew install python@3.12`
- Linux: `sudo apt install python3.12` 或对应包管理器

### 2.2 检查 pdf2zh

用找到的 Python 路径检查 pdf2zh：

```bash
<python_cmd> -c "import pdf2zh; print(pdf2zh.__version__)"
```

如果未安装，自动安装：

```bash
<python_cmd> -m pip install pdf2zh
```

安装完成后告诉用户：「pdf2zh 翻译引擎已安装完成。」

**确定最终使用的命令路径**：
- 如果有 `pdf2zh` 可执行文件在 Scripts/bin 目录下，用它
- 否则用 `<python_cmd> -m pdf2zh` 代替

## 第三步：执行翻译

运行翻译命令：

```bash
PYTHONUTF8=1 <pdf2zh_cmd> "<PDF路径>" -li <源语言> -lo <目标语言> --service <翻译服务>
```

向用户展示进度，等待完成。

## 第四步：验证并报告结果

翻译完成后：

1. 用 PyMuPDF 验证输出 PDF：
   - 页数
   - 矢量图形数量（应与原始一致）
   - 文本长度
   - 是否包含目标语言字符

2. 告诉用户：
   - 输出文件路径（`-mono.pdf` 纯翻译版 和 `-dual.pdf` 双语对照版）
   - 验证结果（图形是否完整、排版是否保持）
   - 已知限制（如有术语保留英文等）

## 参考信息

### 支持的语言代码

| 语言 | 代码 | 语言 | 代码 |
|------|------|------|------|
| 英语 | en | 中文 | zh |
| 日语 | ja | 韩语 | ko |
| 法语 | fr | 德语 | de |
| 西班牙语 | es | 俄语 | ru |

### 支持的翻译服务

| 服务 | 参数 | 需要密钥 | 质量 |
|------|------|---------|------|
| Google Translate | `--service google` | 否 | ★★★ |
| DeepL | `--service deepl` | DEEPL_AUTH_KEY | ★★★★ |
| OpenAI | `--service openai` | OPENAI_API_KEY | ★★★★ |
| Ollama（本地） | `--service ollama` | 否 | ★★★ |

### 已知限制

- 不支持扫描版 PDF（纯图片型，需先 OCR）
- 机器翻译的专业术语可能需人工校对
- 大文件（>50 页）可能耗时较长

### 原理

基于 [PDFMathTranslate (pdf2zh)](https://github.com/Byaidu/PDFMathTranslate)：
1. 提取 PDF 文本和坐标
2. 调用翻译 API
3. 在原始位置替换为翻译文本
4. 所有矢量图形、线条、装饰元素原样保留
