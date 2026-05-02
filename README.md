# arXiv Paper Radar

arXiv Paper Radar 是一个面向科研选题和论文调研的 arXiv 检索工具。输入一个中文或英文研究方向后，它可以自动优化检索关键词，扩大候选论文池，筛选最相关论文，并生成中文总结、关键要点、可参考创新点和发表期刊/会议信息。结果会保存为 JSON，并可通过网页进行筛选和查看。

## 功能

- 根据自然语言研究方向生成 arXiv 检索关键词。
- 从 arXiv 拉取论文标题、论文作者、摘要、类别、链接、发表来源等信息。
- 支持从大候选池中筛选最终论文，例如从 300 篇候选中筛选 30 篇。
- 支持迭代检索：先找一批相关论文，再根据高相关论文扩展下一轮关键词继续检索。
- 使用 LLM 生成中文总结、关键要点、相关性判断和可参考创新点。
- 输出 `publication_venue` 字段，方便判断论文发表在会议、期刊还是仍为空。
- 提供网页查看结果，支持搜索、相关性筛选、主题筛选、类别统计和运行进度查看。

## 文件说明

建议上传到 GitHub 的文件：

- `README.md`：使用说明。
- `index.html`：入口页。
- `arxiv_papers_viewer.html`：结果可视化网页。
- `code/arxiv_paper_research.py`：arXiv 检索与论文分析脚本。
- `.gitignore`：忽略运行生成文件和本地缓存。

默认不建议上传的运行生成文件：

- `arxiv_progress.json`
- `arxiv_raw_papers.json`
- `arxiv_research_results.json`
- `server.log`
- `server.err.log`
- `code/__pycache__/`

这些文件会被 `.gitignore` 忽略，但本地文件不会被删除。

## LLM 配置

脚本使用兼容 Chat Completions 的接口。推荐使用环境变量配置，不要把 API Key 写进代码。

PowerShell 示例：

```powershell
$env:LLM_API_KEY="你的 API key"
$env:LLM_API_URL="https://api.example.com/v1"
$env:LLM_MODEL="your-model-name"
```

如果使用其他兼容服务，替换 `LLM_API_URL` 和 `LLM_MODEL` 即可。

也可以在仓库根目录创建 `.env` 文件，脚本启动时会自动读取：

```text
LLM_API_KEY="你的 API key"
LLM_API_URL="https://api.example.com/v1"
LLM_MODEL="your-model-name"
```

常见环境变量：

- `LLM_API_KEY`：API Key。
- `LLM_API_URL`：接口地址，例如 `https://api.example.com/v1`。
- `LLM_MODEL`：模型名称，例如 `your-model-name`。

如果没有配置 LLM，也可以使用 `--no-llm` 运行。此时脚本只会进行 arXiv 检索和关键词初筛，中文总结与创新点会使用占位内容。

## 快速开始

在仓库根目录运行：

```powershell
python .\code\arxiv_paper_research.py --interest "你的研究方向" --max-results 30 --categories cs.CV,cs.RO,cs.AI
```

## 推荐用法：从 300 篇候选中筛选 30 篇

适合希望尽可能多搜索、再优先筛出最相关工作的场景：

```powershell
python .\code\arxiv_paper_research.py --interest "请检索与你的研究主题最相关的论文，重点关注核心任务、关键方法、应用场景和代表性基线；如果存在明确不相关的方向，请降低优先级。" --candidate-results 300 --max-results 30 --categories cs.CV,cs.AI,cs.CL --refine-rounds 1 --screen-batch-size 30
```

含义：

- `--candidate-results 300`：保留约 300 篇候选论文。
- `--max-results 30`：最终详细分析 30 篇论文。
- `--refine-rounds 1`：根据第一轮高相关论文扩展关键词，再补搜一轮。
- `--screen-batch-size 30`：轻量筛选阶段每批给 LLM 30 篇候选。

## 常用命令

只搜索和关键词初筛，不调用 LLM：

```powershell
python .\code\arxiv_paper_research.py --interest "你的研究方向关键词" --candidate-results 100 --max-results 30 --categories cs.CV,cs.AI --no-llm
```

限制论文提交日期：

```powershell
python .\code\arxiv_paper_research.py --interest "你的研究方向关键词" --candidate-results 200 --max-results 30 --categories cs.CV,cs.AI --from-date 2024-01-01
```

只保留高相关结果：

```powershell
python .\code\arxiv_paper_research.py --interest "你的研究方向关键词" --candidate-results 300 --max-results 30 --categories cs.CV,cs.AI --min-score 60
```

## 参数说明

- `--interest`：检索目标，可以输入中文自然语言、英文关键词或混合描述。
- `--max-results`：最终详细分析并展示的论文数量。
- `--candidate-results`：候选池规模。想从 300 篇里找 30 篇时，设置为 `300`。
- `--per-query-max`：每组 arXiv query 最多抓取多少篇。不填时会根据候选池规模自动估算。
- `--categories`：arXiv 类别，多个类别用英文逗号分隔，例如 `cs.CV,cs.RO,cs.AI`。
- `--from-date`：只检索该日期之后提交的论文，格式 `YYYY-MM-DD`。
- `--to-date`：只检索该日期之前提交的论文，格式 `YYYY-MM-DD`。
- `--sort-by`：arXiv 排序字段，可选 `relevance`、`lastUpdatedDate`、`submittedDate`。
- `--sort-order`：排序方向，可选 `ascending`、`descending`。
- `--request-delay`：arXiv 请求间隔，默认 3 秒。
- `--screen-batch-size`：候选池轻量筛选时，每批交给 LLM 的论文数量。
- `--analysis-batch-size`：详细中文分析时，每批交给 LLM 的论文数量。
- `--min-score`：只保留 LLM 相关性评分不低于该值的论文。
- `--refine-rounds`：迭代扩展检索轮数。`0` 表示不扩展，`1` 通常已经够用。
- `--refine-seed-size`：每轮用于生成下一轮关键词的高相关论文数量。
- `--no-llm`：不调用 LLM。
- `--api-url`：手动指定 LLM API 地址。
- `--api-key`：手动指定 LLM API Key。更推荐使用环境变量。
- `--model`：手动指定 LLM 模型名称。
- `--output`：自定义最终 JSON 输出路径。
- `--raw-output`：自定义原始候选 JSON 输出路径。
- `--progress-output`：自定义运行进度 JSON 输出路径。

## 网页查看

生成结果后，在仓库根目录启动本地静态服务：

```powershell
python -m http.server 8000
```

浏览器打开：

```text
http://localhost:8000/
```

网页支持：

- 查看最终论文卡片。
- 搜索标题、摘要、中文总结和创新点。
- 按相关性分数筛选。
- 按优先级筛选。
- 按主题和 arXiv 类别筛选。
- 查看类别分布。
- 查看运行进度，包括当前检索组、arXiv query、候选筛选批次和 LLM 分析批次。

## 输出文件

脚本默认在仓库根目录生成：

- `arxiv_progress.json`：运行进度。网页会自动轮询该文件。
- `arxiv_raw_papers.json`：候选池原始结果。
- `arxiv_research_results.json`：最终分析结果。

这些文件默认被 `.gitignore` 忽略，不会作为仓库源文件上传。

## 最终 JSON 字段

`arxiv_research_results.json` 的主要结构：

```json
{
  "metadata": {},
  "search_plan": {},
  "candidate_papers": [],
  "papers": []
}
```

其中：

- `metadata`：任务信息、候选池规模、最终结果数量、模型信息等。
- `search_plan`：LLM 生成或 fallback 生成的检索计划。
- `candidate_papers`：候选池论文列表。
- `papers`：最终详细分析后的论文列表。

每篇最终论文包含：

- `arxiv_id`：arXiv ID。
- `title`：标题。
- `authors`：论文作者列表。
- `published`：首次提交时间。
- `updated`：最后更新时间。
- `primary_category`：主类别。
- `categories`：全部 arXiv 类别。
- `abstract`：原始摘要。
- `publication_venue`：发表期刊或会议。若 arXiv 未提供且无法从 comment 中识别，则为空。
- `journal_ref`：arXiv 原始 journal reference。
- `comment`：arXiv comment。
- `links`：arXiv 页面和 PDF 链接。
- `screen_score`：候选池轻量筛选分数。
- `relevance_score`：最终相关性评分。
- `priority`：`high`、`medium` 或 `low`。
- `summary_cn`：中文总结。
- `key_points_cn`：关键要点。
- `innovation_ideas_cn`：可参考创新点。
- `why_relevant_cn`：相关性说明。
- `limitations_cn`：需要进一步确认的限制。
- `matched_topics`：匹配到的主题词。

## 关于发表期刊/会议

`publication_venue` 的来源顺序：

1. 优先使用 arXiv 的 `journal_ref`。
2. 如果 `journal_ref` 为空，尝试从 `comment` 中提取 `Accepted to ...`、`Published in ...`、`To appear in ...` 等信息。
3. 如果仍然无法识别，则保留为空。

arXiv 上很多论文不会填写正式发表来源，因此空值不一定表示论文没有发表，只表示 arXiv 元数据中没有可可靠提取的信息。

## 检索建议

想拿到更相关的结果，建议在 `--interest` 中明确写出：

- 任务名：你要解决的具体问题或评测任务。
- 方法名：你关心的模型、算法、数据表示或训练范式。
- 应用场景：论文需要落到的领域、数据类型或使用环境。
- 对比范围：需要覆盖的代表性基线、经典方法或最新方法。
- 降权内容：你不希望优先出现的方向、数据类型或论文类型。

示例：

```text
我想调研某个研究主题的最相关工作。请优先覆盖核心任务、关键方法、代表性基线、常用数据集和最新高影响力论文；如果论文只包含宽泛背景、与目标任务关系较弱，或属于我明确排除的方向，请降低相关性。
```

## 注意事项

- arXiv API 请求建议保留默认间隔，避免请求过快。
- 大候选池会增加 LLM 调用次数和费用。
- 迭代检索轮数越多，运行时间和 LLM 调用次数越高。
- 发表来源字段依赖 arXiv 元数据，重要论文建议打开 arXiv 页面或 PDF 再人工确认。
