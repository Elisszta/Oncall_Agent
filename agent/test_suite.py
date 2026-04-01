import urllib.request
import urllib.parse
import json
import time
import statistics
import concurrent.futures

BASE_URL = "http://127.0.0.1:8000"

def get_json(path, query_params=None):
    url = f"{BASE_URL}{path}"
    if query_params:
        url += "?" + urllib.parse.urlencode(query_params)
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}

def post_v3_chat(messages):
    url = f"{BASE_URL}/v3/chat"
    data = json.dumps({"messages": messages}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    events = []
    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    content = line[6:]
                    if content == "[DONE]":
                        break
                    try:
                        events.append(json.loads(content))
                    except:
                        pass
        return events
    except Exception as e:
        return [{"event": "error", "detail": str(e)}]

def test_phase1():
    print("=" * 60)
    print("🔍 [TEST] Phase 1: Keyword Search Validation")
    print("=" * 60)
    tests = [
        ("OOM", lambda r: any("sop-001" in item["id"] for item in r.get("results", []))),
        ("故障", lambda r: len(r.get("results", [])) > 0),
        ("replication", lambda r: len(r.get("results", [])) == 0),
        ("CDN", lambda r: any("sop-003" in i["id"] or "sop-010" in i["id"] for i in r.get("results", []))),
        ("&", lambda r: len(r.get("results", [])) > 0),
    ]
    
    results = []
    for q, validator in tests:
        res = get_json("/v1/search", {"q": q})
        passed = validator(res)
        status = "✅ PASS" if passed else "❌ FAIL"
        ids = [item["id"] for item in res.get("results", [])]
        print(f"{status} | Query: '{q}' -> Returns: {ids}")
        results.append({"query": q, "status": status, "returned": ids})
    return results

def test_phase2():
    print("\n" + "=" * 60)
    print("🧠 [TEST] Phase 2: Semantic Search Validation")
    print("=" * 60)
    tests = [
        ("服务器挂了", lambda r: any("sop-001" in item["id"] or "sop-004" in item["id"] for item in r.get("results", [])[:2])),
        ("黑客攻击", lambda r: any("sop-005" in item["id"] for item in r.get("results", [])[:2])),
        ("机器学习模型出问题", lambda r: any("sop-008" in item["id"] for item in r.get("results", [])[:2])),
    ]
    
    results = []
    for q, validator in tests:
        res = get_json("/v2/search", {"q": q})
        passed = validator(res)
        status = "✅ PASS" if passed else "❌ FAIL"
        ids = [item["id"] for item in res.get("results", [])]
        print(f"{status} | Query: '{q}' -> Top Results: {ids[:3]}")
        results.append({"query": q, "status": status, "top_results": ids[:3]})
    return results

def test_phase3():
    print("\n" + "=" * 60)
    print("🤖 [TEST] Phase 3: Agent Tool-Use Validation")
    print("=" * 60)
    
    tests = [
        ("数据库主从延迟超过30秒怎么处理？", "sop-002.html"),
        ("服务 OOM 了怎么办？", "sop-001.html"),
        ("P0 故障的响应流程是什么？", "sop-001.html"), # Actually mention P0 in 001
        ("怀疑有人入侵了系统", "sop-005.html"),
        ("推荐结果质量下降了", "sop-008.html"),
    ]
    
    results = []
    for query, expected_file in tests:
        print(f"Testing Query: '{query}'...")
        events = post_v3_chat([{"role": "user", "content": query}])
        
        found_tool = False
        tool_args = []
        for e in events:
            if e.get("event") == "tool_call":
                found_tool = True
                tool_args.append(e.get("arguments", ""))
        
        passed = any(expected_file in arg for arg in tool_args)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | Expected File: {expected_file} | Tools Called: {tool_args}")
        results.append({"query": query, "status": status, "tools": tool_args, "expected": expected_file})
    return results

def stress_test(endpoint, query, concurrent_users=20, total_requests=100):
    print("\n" + "=" * 60)
    print(f"🚀 [TEST] Stress Testing {endpoint} with '{query}'")
    print("=" * 60)
    
    times = []
    success = 0
    errors = 0
    
    def worker():
        nonlocal success, errors
        start = time.time()
        res = get_json(endpoint, {"q": query})
        t = time.time() - start
        if "error" in res:
            errors += 1
        else:
            success += 1
            times.append(t)
            
    start_total = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        for _ in range(total_requests):
            executor.submit(worker)
            
    total_time = time.time() - start_total
    
    stats = {}
    if times:
        stats = {
            "success": success,
            "total": total_requests,
            "qps": success / total_time,
            "avg_ms": statistics.mean(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }
        print(f"📊 QPS: {stats['qps']:.2f} | Avg: {stats['avg_ms']:.1f}ms")
    return stats

if __name__ == "__main__":
    try:
        urllib.request.urlopen(BASE_URL)
        print(f"🎯 Server {BASE_URL} is running.\n")
        
        p1_res = test_phase1()
        p2_res = test_phase2()
        p3_res = test_phase3()
        
        s1 = stress_test("/v1/search", "故障", concurrent_users=20, total_requests=100)
        s2 = stress_test("/v2/search", "服务器挂了", concurrent_users=5, total_requests=25)
        
        print("\n✅ All tests completed.")
    except Exception as e:
        print(f"⚠️ Error: {e}")
