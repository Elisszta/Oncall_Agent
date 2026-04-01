import os
import json
from openai import AsyncOpenAI
from typing import List, Dict, Any, AsyncGenerator


def read_file(fname: str) -> str:
    """Read a locally stored HTML SOP document by explicit filename."""
    base_dir = os.path.abspath("data")
    file_path = os.path.abspath(os.path.join(base_dir, fname))

    if not file_path.startswith(base_dir) or not os.path.exists(file_path):
        return f"Error: 文件 {fname} 不存在。请使用正确的文件名（如 sop-001.html）。"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Could use BeautifulSoup here to strip scripts, but let's just return HTML/Text directly
            return f.read()
    except Exception as e:
        return f"Error reading file {fname}: {str(e)}"


# The schema sent to OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "readFile",
            "description": "读取并查阅 data 目录下的指定 SOP 文档。这是你唯一获取特定规范流程真相的工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fname": {
                        "type": "string",
                        "description": "要读取的具体 SOP 文件名，例如 'sop-001.html'",
                    }
                },
                "required": ["fname"],
            },
        },
    }
]


async def stream_chat(
    messages: List[Dict[str, str]], config: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    api_key = config.get("api_key")
    if not api_key:
        yield 'data: {"event": "error", "detail": "Missing OpenAI API Key in config"}\n\n'
        return

    client = AsyncOpenAI(api_key=api_key, base_url=config.get("base_url"))
    model = config.get("model", "gpt-4o-mini")
    use_embedding = config.get("use_embedding_search", True)

    system_prompt = (
        "你是负责解答随叫随到 (On-Call) 问题的 AI 智能助手。"
        "你必须调用 `readFile` 工具来查阅具体的 SOP 文档并基于文档内容回答问题，不能凭空捏造架构和数值。"
        "一次可以调用多次工具阅读多个可能的文件。给出处理步骤和排查方向。\n\n"
    )

    try:
        from services.embedding import vec_store

        doc_metadata = vec_store.doc_metadata
        if use_embedding and messages and vec_store.doc_embeddings:
            # We use the embedding phase2 logic implicitly to feed precise hints
            last_msg = messages[-1].get("content", "")
            results = vec_store.search(last_msg, top_k=5)
            if results:
                system_prompt += "根据你的判断，以下文档可能与问题相关，你可以视情况调用 readFile 来获取详情：\n"
                for doc_id, score in results.items():
                    title = doc_metadata.get(doc_id, {}).get("title", "")
                    system_prompt += f"- {doc_id}.html: {title} (相关度: {score:.2f})\n"
        else:
            # Fallback manifest
            system_prompt += (
                "目前系统支持以下 SOP 文档可供查阅（必须带 .html 后缀）：\n"
            )
            if doc_metadata:
                for doc_id, meta in doc_metadata.items():
                    title = meta.get("title", "")
                    system_prompt += f"- {doc_id}.html: {title}\n"
            else:
                system_prompt += "- sop-001.html: 后端服务\n- sop-002.html: 数据库 DBA\n- sop-003.html: 前端\n- sop-004.html: SRE\n- sop-005.html: 安全团队\n- sop-006.html: 数据平台\n- sop-007.html: 移动端\n- sop-008.html: AI & 算法\n- sop-009.html: QA\n- sop-010.html: 网络 & CDN\n"

    except Exception as e:
        system_prompt += "(未能加载索引目录)\n"

    # Ensure system msg is prepended
    internal_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=internal_messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
        )

        tool_calls = {}
        # 1st loop: Initial completion capturing tool reasoning and text
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                # Text answer
                yield f'data: {json.dumps({"event": "message", "delta": delta.content}, ensure_ascii=False)}\n\n'

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "",
                            },
                        }
                    else:
                        if tc.function.arguments:
                            tool_calls[idx]["function"][
                                "arguments"
                            ] += tc.function.arguments

        # Process tools locally and send a 2nd request if needed
        if tool_calls:
            messages_for_second_pass = list(internal_messages)

            # Reconstruct assist message for history compliance
            assist_msg = {"role": "assistant", "tool_calls": []}
            for tc in tool_calls.values():
                assist_msg["tool_calls"].append(
                    {
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                )
            messages_for_second_pass.append(assist_msg)

            # Execution
            for tc in tool_calls.values():
                func_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]

                # Notify frontend with full args
                yield f'data: {json.dumps({"event": "tool_call", "toolName": func_name, "arguments": args_str}, ensure_ascii=False)}\n\n'

                try:
                    args = json.loads(args_str)
                    fname = args.get("fname", "")
                    if func_name == "readFile":
                        output = read_file(fname)
                    else:
                        output = f"Unknown tool: {func_name}"
                except Exception as e:
                    output = str(e)

                yield f'data: {json.dumps({"event": "tool_result", "toolName": func_name}, ensure_ascii=False)}\n\n'

                messages_for_second_pass.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": func_name,
                        "content": output,
                    }
                )

            # 2nd pass
            stream2 = await client.chat.completions.create(
                model=model, messages=messages_for_second_pass, stream=True
            )
            async for chunk in stream2:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f'data: {json.dumps({"event": "message", "delta": delta.content}, ensure_ascii=False)}\n\n'

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f'data: {json.dumps({"event": "error", "detail": str(e)}, ensure_ascii=False)}\n\n'
