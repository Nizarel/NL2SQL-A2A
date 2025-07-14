#!/usr/bin/env python3
"""
Startup script for NL2SQL A2A (Agent-to-Agent) servers
Handles environment detection and provides appropriate URLs for Codespaces, VS Code, and local development
"""

import os
import sys
import asyncio

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

from src.a2a_server import main


def detect_environment():
    """Detect the current environment and return appropriate URLs"""
    # Codespaces detection
    if os.getenv('CODESPACES'):
        codespace_name = os.getenv('CODESPACE_NAME')
        github_codespaces_port_forwarding_domain = os.getenv('GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN', 'preview.app.github.dev')
        
        return {
            'environment': 'Codespaces',
            'base_urls': {
                'orchestrator': f'https://{codespace_name}-8100.{github_codespaces_port_forwarding_domain}',
                'sql_generator': f'https://{codespace_name}-8101.{github_codespaces_port_forwarding_domain}',
                'executor': f'https://{codespace_name}-8102.{github_codespaces_port_forwarding_domain}',
                'summarizer': f'https://{codespace_name}-8103.{github_codespaces_port_forwarding_domain}'
            }
        }
    
    # VS Code detection (when not in Codespaces)
    elif os.getenv('VSCODE_PID') or os.getenv('TERM_PROGRAM') == 'vscode':
        return {
            'environment': 'VS Code',
            'base_urls': {
                'orchestrator': 'http://localhost:8100',
                'sql_generator': 'http://localhost:8101',
                'executor': 'http://localhost:8102',
                'summarizer': 'http://localhost:8103'
            }
        }
    
    # Local development
    else:
        return {
            'environment': 'Local',
            'base_urls': {
                'orchestrator': 'http://localhost:8100',
                'sql_generator': 'http://localhost:8101',
                'executor': 'http://localhost:8102',
                'summarizer': 'http://localhost:8103'
            }
        }


def print_startup_info():
    """Print startup information with environment-specific URLs"""
    env_info = detect_environment()
    environment = env_info['environment']
    base_urls = env_info['base_urls']
    
    print("🚀 NL2SQL Multi-Agent A2A (Agent-to-Agent) Server")
    print("=" * 60)
    print(f"🌍 Environment: {environment}")
    print()
    
    print("📡 A2A Agent Endpoints:")
    print("-" * 30)
    
    for agent_type, base_url in base_urls.items():
        print(f"🤖 {agent_type.replace('_', ' ').title()} Agent:")
        print(f"   • Agent Card: {base_url}/.well-known/agent.json")
        print(f"   • A2A Endpoint: {base_url}/")
        print(f"   • Documentation: {base_url}/docs")
        print()
    
    print("💡 Usage Examples:")
    print("-" * 20)
    print("1. Test Agent Card:")
    print(f"   curl {base_urls['orchestrator']}/.well-known/agent.json")
    print()
    print("2. Send A2A Message (JSON-RPC):")
    print(f"   curl -X POST {base_urls['orchestrator']}/ \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{")
    print('       "jsonrpc": "2.0",')
    print('       "method": "message/send",')
    print('       "id": "1",')
    print('       "params": {')
    print('         "message": {')
    print('           "role": "user",')
    print('           "content": "Show me the top 10 customers by revenue"')
    print('         }')
    print('       }')
    print("     }'")
    print()
    print("3. Agent Communication Workflow:")
    print("   • Send question to Orchestrator → Complete pipeline")
    print("   • Send question to SQL Generator → Get SQL query")
    print("   • Send SQL to Executor → Get results")
    print("   • Send results to Summarizer → Get insights")
    print()
    
    if environment == 'Codespaces':
        print("🔧 Codespaces Port Configuration:")
        print("   Make sure ports 8100-8103 are forwarded and set to 'Public'")
        print("   in the Ports tab of VS Code for external access.")
        print()


if __name__ == "__main__":
    print_startup_info()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  A2A servers stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting A2A servers: {e}")
        sys.exit(1)
