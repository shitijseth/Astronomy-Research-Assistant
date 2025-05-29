# ============================
# astronomy_assistant_final.py
# Clean, complete astronomy assistant with MCP tools and multi-LLM support
# ============================

import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from contextlib import AsyncExitStack

import requests

# Correct MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Try importing various LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Configuration
@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000

# Abstract base class for LLM providers
class LLMProvider(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]], tools_description: str = "") -> str:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass

# OpenAI Provider
class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.api_base
        )
    
    async def generate_response(self, messages: List[Dict[str, str]], tools_description: str = "") -> str:
        try:
            system_msg = {
                "role": "system",
                "content": f"""You are an advanced astronomy research assistant with access to various astronomical tools and databases.

Available tools and their capabilities:
{tools_description}

Please provide detailed, accurate responses and suggest using specific tools when appropriate for the user's query."""
            }
            
            full_messages = [system_msg] + messages
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=full_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "OpenAI",
            "model": self.config.model,
            "type": "Cloud API"
        }

# Anthropic Claude Provider
class AnthropicProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
        )
    
    async def generate_response(self, messages: List[Dict[str, str]], tools_description: str = "") -> str:
        try:
            system_prompt = f"""You are an advanced astronomy research assistant with access to various astronomical tools and databases.

Available tools and their capabilities:
{tools_description}

Please provide detailed, accurate responses and suggest using specific tools when appropriate for the user's query."""

            claude_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    continue
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            response = await self.client.messages.create(
                model=self.config.model,
                system=system_prompt,
                messages=claude_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Anthropic",
            "model": self.config.model,
            "type": "Cloud API"
        }

# HuggingFace Provider (local inference)
class HuggingFaceProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace transformers not installed. Run: pip install transformers torch")
        
        print(f"Loading HuggingFace model: {config.model}")
        print("This may take a few minutes on first load...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        
        try:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                trust_remote_code=True,
                use_fast=True
            )
            
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
                print(f"✅ Model loaded on GPU with FP16 optimization")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print(f"✅ Model loaded on CPU")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    async def generate_response(self, messages: List[Dict[str, str]], tools_description: str = "") -> str:
        try:
            prompt = f"""You are a helpful astronomy assistant. Answer questions clearly and concisely.

Available tools:
{tools_description}

"""
            recent_messages = messages[-3:] if len(messages) > 3 else messages
            
            for msg in recent_messages:
                role = msg["role"].title()
                prompt += f"{role}: {msg['content']}\n"
            
            prompt += "Assistant:"
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=False
            )
            
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(self.config.max_tokens, 200),
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            # Clean up response
            response_lines = response.split('\n')
            clean_response = []
            
            for line in response_lines:
                if any(phrase in line.lower() for phrase in [
                    'in your role as', 'you have been tasked', 'here are some clues',
                    'question:', 'from clue', 'looking at clue'
                ]):
                    break
                clean_response.append(line)
            
            final_response = '\n'.join(clean_response).strip()
            
            if len(final_response) < 10:
                final_response = response.split('.')[0] + '.' if '.' in response else response
            
            return final_response
            
        except Exception as e:
            return f"HuggingFace inference error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        device = "GPU" if torch.cuda.is_available() and next(self.model.parameters()).is_cuda else "CPU"
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_info = f" ({torch.cuda.get_device_name(0)})"
        
        return {
            "provider": "HuggingFace",
            "model": self.config.model,
            "type": "Local",
            "device": device + gpu_info,
            "precision": "FP16" if torch.cuda.is_available() else "FP32"
        }

# Main Assistant Class
class AstronomyAssistant:
    def __init__(self):
        self.providers = {}
        self.current_provider = None
        self.tools = []
        self.conversation_history = []
        self.session = None
        self.exit_stack = None
    
    async def initialize(self):
        """Initialize MCP connection and load available tools."""
        try:
            print("🔌 [MCP] Connecting to astronomy server...")
            
            server_params = StdioServerParameters(
                command="python",
                args=["astronomy_mcp_server.py"]
            )
            
            self.exit_stack = AsyncExitStack()
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            print("🤝 [MCP] Initializing session...")
            await self.session.initialize()
            
            print("🛠️ [MCP] Loading available tools...")
            tools_result = await self.session.list_tools()
            self.tools = tools_result.tools if tools_result else []
            
            print(f"✅ [MCP] Connected successfully!")
            print(f"📊 [MCP] {len(self.tools)} tools available:")
            for i, tool in enumerate(self.tools, 1):
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                print(f"   {i}. {tool_name}")
            
        except Exception as e:
            print(f"❌ [MCP] Connection failed: {e}")
            print("⚠️ [MCP] Running without tool access.")
    
    async def cleanup(self):
        """Clean up MCP connections."""
        if self.exit_stack:
            await self.exit_stack.aclose()
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute an MCP tool."""
        if not hasattr(self, 'session') or not self.session:
            return "MCP session not available"
        
        try:
            print(f"🔧 [TOOL] Executing: {tool_name}")
            print(f"📝 [TOOL] Parameters: {parameters}")
            
            result = await self.session.call_tool(tool_name, parameters)
            
            if result.content:
                if hasattr(result.content[0], 'text'):
                    output = result.content[0].text
                else:
                    output = str(result.content[0])
                
                print(f"✅ [TOOL] Success: {tool_name}")
                print(f"📊 [TOOL] Output length: {len(output)} characters")
                return output
            else:
                print(f"⚠️ [TOOL] No output returned from {tool_name}")
                return "No output returned"
                
        except Exception as e:
            print(f"❌ [TOOL] Error executing {tool_name}: {str(e)}")
            return f"Tool execution error: {str(e)}"
    
    def add_provider(self, name: str, config: LLMConfig):
        """Add an LLM provider."""
        try:
            if config.provider.lower() == "openai":
                self.providers[name] = OpenAIProvider(config)
            elif config.provider.lower() == "anthropic":
                self.providers[name] = AnthropicProvider(config)
            elif config.provider.lower() == "huggingface":
                self.providers[name] = HuggingFaceProvider(config)
            else:
                raise ValueError(f"Unknown provider: {config.provider}")
            
            print(f"[INFO] Added {config.provider} provider: {name}")
            
            if self.current_provider is None:
                self.current_provider = name
                
        except Exception as e:
            print(f"[ERROR] Failed to add provider {name}: {e}")
    
    def switch_provider(self, name: str) -> bool:
        """Switch to a different LLM provider."""
        if name in self.providers:
            self.current_provider = name
            return True
        return False
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all available providers."""
        return [
            {
                "name": name,
                "info": provider.get_model_info(),
                "active": name == self.current_provider
            }
            for name, provider in self.providers.items()
        ]
    
    def get_tools_description(self) -> str:
        """Generate description of available tools."""
        if not self.tools:
            return "No tools available"
        
        description = "Available astronomy tools:\n"
        for tool in self.tools:
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            tool_desc = tool.description if hasattr(tool, 'description') else 'No description'
            description += f"- {tool_name}: {tool_desc}\n"
        
        return description
    
    async def analyze_and_use_tools(self, message: str) -> str:
        """Analyze message and automatically use relevant tools."""
        if not hasattr(self, 'session') or not self.session or not self.tools:
            print(f"⚠️ [ANALYSIS] No MCP session available - tools disabled")
            return ""
        
        message_lower = message.lower()
        tool_results = []
        
        print(f"🔍 [ANALYSIS] Checking if any tools are relevant...")
        
        # Check for basic astronomy searches
        if any(word in message_lower for word in ['distance', 'sun', 'earth', 'solar system', 'light years']):
            print(f"🎯 [ANALYSIS] Detected basic astronomy question")
            result = await self.execute_tool("search_basic_astronomy", {"query": message})
            tool_results.append(f"Astronomy Info: {result}")
        
        # Check for APOD requests
        elif any(word in message_lower for word in ['apod', 'astronomy picture', 'picture of the day', 'nasa picture']):
            print(f"🎯 [ANALYSIS] Detected APOD request")
            result = await self.execute_tool("get_apod", {})
            tool_results.append(f"NASA APOD: {result}")
        
        # Check for ISS requests
        elif any(word in message_lower for word in ['iss', 'international space station', 'space station location']):
            print(f"🎯 [ANALYSIS] Detected ISS request")
            result = await self.execute_tool("get_iss_location", {})
            tool_results.append(f"ISS Location: {result}")
        
        if tool_results:
            print(f"✅ [ANALYSIS] Used {len(tool_results)} tools automatically")
            return "\n\n".join(tool_results)
        else:
            print(f"💭 [ANALYSIS] No relevant tools found, using LLM knowledge only")
            return ""
    
    async def chat_with_tools(self, message: str) -> str:
        """Enhanced chat that can automatically use tools when relevant."""
        if not self.current_provider:
            return "No LLM provider configured. Please add a provider first."
        
        # Add message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Check if we should use any tools based on the message
        tool_results = await self.analyze_and_use_tools(message)
        
        # Get current provider
        provider = self.providers[self.current_provider]
        
        # Generate response with tool context
        tools_desc = self.get_tools_description()
        if tool_results:
            tools_desc += f"\n\nRecent tool results:\n{tool_results}"
        
        response = await provider.generate_response(
            self.conversation_history[-10:],
            tools_desc
        )
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def save_conversation(self, filename: str):
        """Save conversation to file."""
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "provider": self.current_provider,
                "conversation": self.conversation_history
            }, f, indent=2)
    
    def load_conversation(self, filename: str):
        """Load conversation from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
            self.conversation_history = data.get("conversation", [])

# Command-line interface
async def main():
    assistant = AstronomyAssistant()
    
    try:
        await assistant.initialize()
        
        print("\n=== Multi-LLM Astronomy Assistant ===")
        print("Setting up available providers...\n")
        
        # Setup providers
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            assistant.add_provider("gpt-4", LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.7
            ))
        
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            assistant.add_provider("claude-3", LLMConfig(
                provider="anthropic",
                model="claude-3-sonnet-20240229",
                temperature=0.7
            ))
        
        if len(assistant.providers) == 0 and HUGGINGFACE_AVAILABLE:
            print("[INFO] No API providers available, setting up local HuggingFace model...")
            assistant.add_provider("phi-2", LLMConfig(
                provider="huggingface",
                model="microsoft/phi-2",
                temperature=0.7,
                max_tokens=512
            ))
        
        if len(assistant.providers) == 0:
            print("[ERROR] No LLM providers available!")
            return
        
        # Show available providers
        providers = assistant.list_providers()
        print("Available providers:")
        for i, provider in enumerate(providers):
            status = "🟢 ACTIVE" if provider["active"] else "⚪"
            print(f"{i+1}. {status} {provider['name']} ({provider['info']['provider']} - {provider['info']['model']})")
        
        print(f"\nUsing: {assistant.current_provider}")
        print("\nCommands:")
        print("- 'switch <number>' - Switch LLM provider")
        print("- 'providers' - List available providers") 
        print("- 'tools' - List available tools")
        print("- 'clear' - Clear conversation history")
        print("- 'exit' - Quit")
        print("\nAsk me anything about astronomy! 🌟\n")
        
        while True:
            try:
                user_input = input(f"[{assistant.current_provider}] You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'providers':
                    providers = assistant.list_providers()
                    for i, provider in enumerate(providers):
                        status = "🟢 ACTIVE" if provider["active"] else "⚪"
                        print(f"{i+1}. {status} {provider['name']} ({provider['info']['provider']})")
                    continue
                elif user_input.lower().startswith('switch '):
                    try:
                        idx = int(user_input.split()[1]) - 1
                        providers = assistant.list_providers()
                        if 0 <= idx < len(providers):
                            new_provider = providers[idx]["name"]
                            assistant.switch_provider(new_provider)
                            print(f"Switched to: {new_provider}")
                        else:
                            print("Invalid provider number")
                    except (ValueError, IndexError):
                        print("Usage: switch <number>")
                    continue
                elif user_input.lower() == 'tools':
                    if assistant.tools:
                        print(f"\n🛠️ Available Tools ({len(assistant.tools)} total):")
                        print("=" * 50)
                        for i, tool in enumerate(assistant.tools, 1):
                            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                            tool_desc = tool.description if hasattr(tool, 'description') else 'No description'
                            print(f"{i:2d}. 🔧 {tool_name}")
                            print(f"     📝 {tool_desc}")
                            print()
                        print("💡 Tools are automatically used when relevant to your questions!")
                    else:
                        print("❌ No tools available (MCP connection failed)")
                    continue
                elif user_input.lower() == 'clear':
                    assistant.clear_history()
                    print("Conversation history cleared")
                    continue
                
                # Process regular chat message
                print(f"\n💬 [CHAT] Processing your question...")
                response = await assistant.chat_with_tools(user_input)
                print(f"\n🤖 Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye! 🚀")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    finally:
        await assistant.cleanup()

if __name__ == "__main__":
    asyncio.run(main())