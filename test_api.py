#!/usr/bin/env python3
import httpx
import sys
import json
from typing import Optional

def test_health():
    try:
        r = httpx.get("http://localhost:8000/health", timeout=5.0)
        r.raise_for_status()
        print(f"✅ Healthy: {json.dumps(r.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def query_api(query: str, conversation_id: Optional[str] = None):
    print(f"\n📝 Query: {query}\n⏳ Waiting...\n")
    try:
        payload = {"query": query}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        r = httpx.post("http://localhost:8000/query", json=payload, timeout=120.0)
        r.raise_for_status()
        result = r.json()
        
        print("="*80)
        print(f"{result['response']}")
        print("="*80)
        print(f"\nSources: {', '.join(result['sources_used'])}")
        print(f"Time: {result.get('execution_time', 0):.2f}s")
        return result
    except httpx.HTTPStatusError as e:
        print(f"❌ Error {e.response.status_code}: {e.response.text}")
    except httpx.TimeoutException:
        print("❌ Timeout")
    except Exception as e:
        print(f"❌ Error: {e}")
    return None

def main():
    print("\n" + "="*80 + "\nHealthcare Agent API Test\n" + "="*80 + "\n")
    
    if not test_health():
        print("\n⚠️  Services not running. Start with: ./start-services.sh")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the latest treatments for type 2 diabetes?"
    result = query_api(query)
    
    if result:
        print("\n" + "="*80 + "\n💬 Multi-turn test\n" + "="*80)
        query_api("What are the side effects?", conversation_id=result['conversation_id'])

if __name__ == "__main__":
    main()
