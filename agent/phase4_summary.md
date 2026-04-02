# Phase 4: ReAct 智能排障 Agent 逻辑总结

Phase 4 是该随叫随到 (On-Call) 助手的核心演进，引入了基于 **ReAct (Reasoning and Acting)** 框架的自主排障闭环。Agent 不再仅仅是简单的问答或搜索，而是能够根据用户问题，自主决定搜索哪些文档、查阅哪些细节，并最终整合出高度准确的排障建议。

## 1. 核心架构：ReAct 推理循环

在 `agent/services/agent.py` 中，`react_stream_chat` 函数实现了标准的 **Thought -> Action -> Observation** 循环：

1.  **Thought (推理)**：模型根据当前对话上下文，思考下一步该做什么（如：“我需要查看关于 CPU 飙升的 SOP 文档”）。
2.  **Action (动作)**：模型通过 **Function Calling** 调用预定义的工具。
3.  **Observation (观察)**：系统执行工具并获取结果（如 SOP 的具体文本内容），将其反馈给模型。
4.  **循环终止**：当模型认为已搜集足够信息，它会输出最终解答。该循环在 `react_stream_chat` 中最多执行 **6 次** 以防止死循环或过度消耗 Token。

---

## 2. 工具集 (Tools)

Agent 拥有 4 个核心工具，用于按需获取知识：

-   `searchSOP`: **向量搜索**。通过关键词寻找最匹配的文档 ID（如 `sop-001.html`）。
-   `listSOP`: **清单查询**。列出所有可用的 SOP 文档标题。
-   `readFile`: **全文读取**。直接获取特定 HTML 文档的原始内容。
-   `getSummary`: **精准摘要**。在大型文档中，仅提取与特定查询最相关的分块 (Chunks)，有效节省 Token 并提高针对性。

---

## 3. 抗幻觉机制 (Hallucination Interception)

为了解决 LLM 容易“编造”不存在的文件名的问题，Phase 4 实现了 `_intercept_hallucination` 逻辑：

-   **强制检查**：在 `readFile` 或 `getSummary` 执行前，系统会校验文件名是否存在。
-   **模糊匹配建议**：如果模型输入的文件名不存在，系统通过**向量相似度搜索**寻找最接近的真实文件名，并作为 Error 信息反馈给 Agent（如：“文件 xxx 不存在，你是不是要找 yyy？”）。Agent 收到报错后通常会自动纠正并重新调用。

---

## 4. 系统启动与预搜索 (System Pre-search)

在进入 ReAct 循环之前，系统会执行一个隐式的预处理步骤：
-   **向量预搜**：提取用户最后一条提问，在向量库中寻找 Top 5 相关的文档。
-   **上下文引导**：将这些“高度相关文档”的名称和标题直接写入 System Prompt 的末尾，显著降低了 Agent 第一次“盲目尝试”的概率，提高了首轮推理的准确性。

---

## 5. 流式通信协议 (SSE Event Stream)

后端采用 Server-Sent Events (SSE) 协议与前端通信，实时体现 Agent 的思考过程：

-   `event: thought`: 传输 Agent 的推理文本片段。
-   `event: tool_call`: 通知前端 Agent 决定调用哪个工具及其参数。
-   `event: tool_result`: 工具执行完毕后的摘要反馈。
-   `event: message_finalized`: 最终回答生成完毕。
-   `event: error` / `event: loop_limit`: 异常或达到步数限制的提示。

---

## 6. 技术栈与集成

-   **LLM API**: 默认使用 OpenAI / GPT-4o-mini 或兼容接口（如 DeepSeek）。
-   **向量模型**: `BAAI/bge-small-zh-v1.5`（本地加载）。
-   **后端**: FastAPI `StreamingResponse`。
-   **前端**: 独立的 `v4/index.html` 模块，包含专门的“思考过程”动态展示框。
