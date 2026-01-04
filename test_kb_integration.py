"""
Test script for Knowledge Base Integration
Demonstrates how the LLM queries KB and uses context
"""
import sys
sys.path.insert(0, '.')

from app.modules.llm.receptionist_llm import ReceptionistLLM

print("=" * 70)
print("KNOWLEDGE BASE INTEGRATION TEST")
print("=" * 70)

# Initialize LLM with KB integration
print("\n1. Initializing LLM with KB integration...")
llm = ReceptionistLLM(
    model_name="microsoft/DialoGPT-medium",
    api_key="test_key",  # Replace with actual key
    kb_service_url="http://localhost:8001"
)
print("   [OK] LLM initialized with KB URL:", llm.kb_service_url)

# Test prompt building without KB
print("\n2. Testing prompt WITHOUT KB context...")
prompt_no_kb = llm._build_prompt("What products do you sell?")
print("\n   Generated Prompt (no KB):")
print("   " + "-" * 66)
print("   " + prompt_no_kb.replace("\n", "\n   "))
print("   " + "-" * 66)

# Test prompt building with KB context (simulated)
print("\n3. Testing prompt WITH KB context (simulated)...")
kb_context = """You are a customer service agent for TechStore.

Relevant Information:

[Source 1]
We sell electronics including laptops (Dell XPS 15, HP Spectre, Lenovo ThinkPad), smartphones (iPhone 15, Samsung Galaxy S24), tablets, and gaming consoles.

[Source 2]
Our product categories include: Laptops, Smartphones, Tablets, Gaming Consoles, Audio Equipment, and Accessories.

[Source 3]
All products come with manufacturer warranty. We offer free shipping on orders over $50.
"""

prompt_with_kb = llm._build_prompt("What products do you sell?", kb_context=kb_context)
print("\n   Generated Prompt (with KB):")
print("   " + "-" * 66)
print("   " + prompt_with_kb.replace("\n", "\n   "))
print("   " + "-" * 66)

# Show the difference
print("\n4. COMPARISON:")
print(f"   Prompt length WITHOUT KB: {len(prompt_no_kb)} characters")
print(f"   Prompt length WITH KB:    {len(prompt_with_kb)} characters")
print(f"   KB adds {len(prompt_with_kb) - len(prompt_no_kb)} characters of context")

print("\n5. How it works in production:")
print("   " + "=" * 66)
print("   a) User asks: 'What products do you sell?'")
print("   b) If company_id provided:")
print("      - Query KB: POST /api/v1/context")
print("      - Retrieve top 3 relevant chunks")
print("      - Inject into prompt (as shown above)")
print("   c) LLM generates response WITH company-specific context")
print("   d) Result: Accurate answer about TechStore products")
print("   " + "=" * 66)

print("\n6. Testing KB query method (will fail if KB not running)...")
try:
    kb_result = llm._query_knowledge_base(
        query="What products do you sell?",
        company_id="techstore"
    )
    if kb_result:
        print("   [OK] KB query successful!")
        print(f"   Context length: {len(kb_result)} characters")
    else:
        print("   [WARN] KB returned no context (confidence too low or empty)")
except Exception as e:
    print(f"   [EXPECTED] KB service unavailable: {str(e)}")
    print("   This is normal if KB service isn't running")

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETE")
print("=" * 70)
print("\nTo see this in action:")
print("1. Start KB service: cd knowledge-base-system && python run.py")
print("2. Load company: curl -X POST http://localhost:8001/api/v1/company/load \\")
print("                      -d '{\"company_id\": \"techstore\"}'")
print("3. Start voice agent: python -m app.main")
print("4. Test: curl -X POST http://localhost:8000/chat \\")
print("              -d '{\"message\": \"What do you sell?\", \"company_id\": \"techstore\"}'")
print("=" * 70)
