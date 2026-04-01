import os
import json
from openai import AsyncOpenAI
from typing import List, Dict, Any, AsyncGenerator
import numpy as np

def _intercept_hallucination(fname: str) -> str:
    from services.embedding import vec_store
    known_files = set(vec_store.doc_metadata.keys())
    base = fname.replace(".html", "")
    if base not in known_files:
        if vec_store.doc_embeddings:
            results = vec_store.search(fname, top_k=1)
            if results:
                best = max(results.items(), key=lambda x: x[1])[0]
                return f"[INTERCEPT] 文件 {fname} 不存在。最相似的文档是 {best}.html，建议尝试调整参数调用。"
        return f"Error: 文件 {fname} 不存在。"
    return ""

def safe_read_file(fname: str) -> str:
    """Read a locally stored HTML SOP document with hallucination interception."""
    intercept_msg = _intercept_hallucination(fname)
    if intercept_msg:
        return intercept_msg
    
    base = fname.replace(".html", "")
    # Read the file content
    try:
        base_dir = os.path.abspath("data")
        file_path = os.path.abspath(os.path.join(base_dir, f"{base}.html"))
        if not file_path.startswith(base_dir) or not os.path.exists(file_path):
            return f"Error: 文件 {fname} 不存在。"
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {fname}: {str(e)}"

def search_sop(query: str) -> str:
    from services.embedding import vec_store
    if not vec_store.doc_embeddings:
        return "索引未初始化或无可用文档"
    results = vec_store.search(query, top_k=5)
    if not results:
        return "没有找到匹配的SOP文档。"
    res_str = "搜索结果: \n"
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for doc_id, score in sorted_results:
        title = vec_store.doc_metadata.get(doc_id, {}).get("title", "")
        res_str += f"- {doc_id}.html: {title} (匹配度: {score:.4f})\n"
    return res_str

def list_sop() -> str:
    from services.embedding import vec_store
    if not vec_store.doc_metadata:
        return "索引未初始化或无可用文档"
    res_str = "可用SOP清单:\n"
    for doc_id, meta in vec_store.doc_metadata.items():
        title = meta.get("title", "")
        res_str += f"- {doc_id}.html: {title}\n"
    return res_str

def get_summary(fname: str, query: str) -> str:
    from services.embedding import vec_store, get_model
    
    intercept_msg = _intercept_hallucination(fname)
    if intercept_msg:
        return intercept_msg
        
    base = fname.replace(".html", "")
        
    chunks = vec_store.doc_chunks.get(base)
    if not chunks:
        return "文件无分块数据"
        
    model = get_model()
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    chunk_embs = vec_store.doc_embeddings.get(base)
    
    if chunk_embs is None:
        return "嵌入丢失"
        
    similarities = np.dot(chunk_embs, query_emb)
    top_k = min(3, len(similarities))
    # Add defensive check for 0-length similarities
    if len(similarities) == 0:
         return "无内容可总结"

    top_k_idx = similarities.argsort()[-top_k:][::-1]
    
    res = f"文件 {fname} 与查询相关的片段(用于避免溢出和精确查找):\n"
    for i, idx in enumerate(top_k_idx):
        res += f"----片段 {i+1} (相似度: {similarities[idx]:.4f})----\n{chunks[idx]}\n\n"
        
    return res

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "searchSOP",
            "description": "通过问题关键词搜索最相关的SOP文档，返回文档名及评分。可以用于缩小查找范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或具体问题的概要",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "readFile",
            "description": "读取完整的SOP文档内容。当明确知道文档名称并需要全量信息时使用。注意：不要用于过大的文档！",
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
    },
    {
        "type": "function",
        "function": {
            "name": "listSOP",
            "description": "列出系统中所有可用的 SOP 文档的清单与标题。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getSummary",
            "description": "读取特定SOP文档，并且只提取和返回与你提出问题最相关的文本片段(防Token溢出)。推荐在文档较大时使用此工具查阅特定细节。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fname": {
                        "type": "string",
                        "description": "文档名称，如 'sop-002.html'",
                    },
                    "query": {
                        "type": "string",
                        "description": "你希望在此文档中寻找的具体问题或关键词",
                    }
                },
                "required": ["fname", "query"],
            },
        },
    }
]

async def react_stream_chat(
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
        "你是负责解答随叫随到 (On-Call) 问题的 AI 智能排障小助手。\n"
        "你可以通过工具（searchSOP, listSOP, readFile, getSummary）查阅 SOP 文档来寻找正确排障流程与根因解释。\n"
        "回答原则：\n"
        "1. 每当你需要搜集信息时，请果断调用工具。每次请求你可以调用一个或多个工具。\n"
        "2. 在调用工具前，你生成的任何文本都将被视为你的**排障推理步骤 (Thought)**。\n"
        "3. 当你获取完所有必要信息，准备输出发给用户的 **最终解答** 时，请务必直接输出最终答案（不再调用工具），此时你输出的文本将被视为最终解答。\n"
        "禁止凭空想象 SOP 内容，不要捏造未在文档中提及的阈值、命令与架构。\n"
    )

    try:
        from services.embedding import vec_store
        if use_embedding and messages and vec_store.doc_embeddings:
            last_msg = messages[-1].get("content", "")
            if isinstance(last_msg, str):
                results = vec_store.search(last_msg, top_k=5)
                if results:
                    system_prompt += "\n\n【系统预搜索提示】根据用户的问题，向量系统发现以下文档可能高度相关，你可以直接调用工具查阅这些文档：\n"
                    for doc_id, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
                        title = vec_store.doc_metadata.get(doc_id, {}).get("title", "")
                        system_prompt += f"- {doc_id}.html: {title}\n"
    except Exception as e:
        system_prompt += "\n(向量索引未加载)\n"

    internal_messages = [{"role": "system", "content": system_prompt}] + messages

    MAX_LOOPS = 6

    for loop_count in range(MAX_LOOPS):
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=internal_messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
            )

            tool_calls = {}
            has_tool_call = False
            
            # Since we must stream, we will just stream everything as 'thought'. If has_tool_call is False,
            # that means the model decided to answer! We could have streamed it as 'message' if we knew!
            # Since OpenAI streams content before tool_calls, we don't know if it will call a tool until the end!
            
            assist_content = ""

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                if delta.content:
                    assist_content += delta.content
                    yield f'data: {json.dumps({"event": "thought", "delta": delta.content}, ensure_ascii=False)}\n\n'
                
                if delta.tool_calls:
                    has_tool_call = True
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or "",
                                }
                            }
                        else:
                            if hasattr(tc, "id") and tc.id:
                                tool_calls[idx]["id"] = tc.id
                            if hasattr(tc, "type") and tc.type:
                                tool_calls[idx]["type"] = tc.type
                            if tc.function.arguments:
                                tool_calls[idx]["function"]["arguments"] += tc.function.arguments

                stop_reason = chunk.choices[0].finish_reason
                if stop_reason in ("stop", "length", "content_filter"):
                    if not has_tool_call:
                        # Model finished its text response (normally, truncated, or filtered).
                        # Whatever was streamed as 'thought' is the final answer.
                        yield f'data: {json.dumps({"event": "message_finalized", "delta": ""})}\n\n'
                        if stop_reason == "length":
                            yield f'data: {json.dumps({"event": "error", "detail": "⚠️ 模型回复因达到 Token 上限而被截断，内容可能不完整。"})}\n\n'
                        elif stop_reason == "content_filter":
                            yield f'data: {json.dumps({"event": "error", "detail": "⚠️ 模型回复被安全过滤器拦截，无法显示完整内容。"})}\n\n'
                        yield "data: [DONE]\n\n"
                        return # Exit generator!

            # Process tools if any
            if tool_calls:
                # Use None for content when only tool calls are present (more API-correct than empty string)
                assist_msg = {"role": "assistant", "tool_calls": [], "content": assist_content or None}
                for tc in tool_calls.values():
                    assist_msg["tool_calls"].append(
                        {
                            "id": tc["id"],
                            "type": tc.get("type") or "function",  # defensive: type is only on first delta chunk
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                    )
                internal_messages.append(assist_msg)

                for tc in tool_calls.values():
                    func_name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]

                    yield f'data: {json.dumps({"event": "tool_call", "toolName": func_name, "arguments": args_str}, ensure_ascii=False)}\n\n'

                    try:
                        args = json.loads(args_str)
                        if func_name == "readFile":
                            output = safe_read_file(args.get("fname", ""))
                        elif func_name == "getSummary":
                            output = get_summary(args.get("fname", ""), args.get("query", ""))
                        elif func_name == "searchSOP":
                            output = search_sop(args.get("query", ""))
                        elif func_name == "listSOP":
                            output = list_sop()
                        else:
                            output = f"Unknown tool: {func_name}"
                    except Exception as e:
                        output = str(e)

                    yield f'data: {json.dumps({"event": "tool_result", "toolName": func_name, "summary": output[:200] + "..." if len(output) > 200 else output}, ensure_ascii=False)}\n\n'

                    internal_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": func_name,
                            "content": output,
                        }
                    )
            else:
                # Edge case, finish without stop but without tools?
                break

        except Exception as e:
            yield f'data: {json.dumps({"event": "error", "detail": str(e)}, ensure_ascii=False)}\n\n'
            return

    yield f'data: {json.dumps({"event": "loop_limit", "detail": "Reached maximum reasoning steps (6). Please start a new query."}, ensure_ascii=False)}\n\n'
    yield "data: [DONE]\n\n"