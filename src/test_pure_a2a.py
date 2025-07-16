"""Test the pure A2A server components"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pure_a2a_server import get_agent_card
    
    print("🧪 Testing Pure A2A Server Components...")
    
    # Test agent card creation
    card = get_agent_card('localhost', 8002)
    print(f"✅ Agent Card: {card.name} v{card.version}")
    print(f"✅ Capabilities: streaming={card.capabilities.streaming}")
    print(f"✅ Skills: {len(card.skills)} skill(s) defined")
    print(f"✅ Examples: {len(card.skills[0].examples)} example(s)")
    
    print("\n🎉 Pure A2A Server components test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
