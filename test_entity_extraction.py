"""Test entity extraction with Claude 4.5 to see actual LLM calls."""
import asyncio
import litellm
import os

# Enable debug logging
litellm.set_verbose = True

async def test_extraction():
    # Load the extraction prompt
    with open("prompts/extract_graph.txt", "r") as f:
        prompt_template = f.read()

    # Sample text from A Christmas Carol
    test_text = """Ebenezer Scrooge is a covetous old man who is the surviving partner
    of the firm of Scrooge and Marley. Bob Cratchit is his clerk. Scrooge's nephew Fred
    is a kind-hearted man."""

    # Format the prompt
    entity_types = "organization,person,geo,event"
    prompt = prompt_template.replace("{entity_types}", entity_types).replace("{input_text}", test_text)

    print("=" * 80)
    print("PROMPT FIRST 500 CHARS:")
    print("=" * 80)
    print(prompt[:500])
    print("=" * 80)

    print("\n" + "=" * 80)
    print("CALLING LLM (litellm.acompletion)...")
    print("=" * 80)

    # Call directly with litellm (exactly as GraphRAG does)
    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("GRAPHRAG_API_KEY"),
        temperature=0,
        max_tokens=4096,
        drop_params=True,  # Drop unsupported params
    )

    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response.choices[0].message.content)
    print("=" * 80)

    # Check if it has delimiter format
    content = response.choices[0].message.content
    if "<|>" in content and "##" in content:
        print("\n✅ SUCCESS: Response contains delimiter format (<|> and ##)")
    else:
        print("\n❌ FAILED: Response does NOT contain delimiter format")
        print("   Checking if it's JSON instead...")
        import json
        try:
            json.loads(content)
            print("   ⚠️  Response is JSON format (not delimiter format)")
        except:
            print("   Response is neither delimiter nor valid JSON")

if __name__ == "__main__":
    asyncio.run(test_extraction())
