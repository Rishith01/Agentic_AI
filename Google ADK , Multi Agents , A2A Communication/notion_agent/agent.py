import os
import json
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from mcp import StdioServerParameters
from config import NOTION_API_KEY

# SambaNova API Key (Get it free at cloud.sambanova.ai)
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

def create_notion_agent() -> LlmAgent:
    return LlmAgent(
        name="notion_agent_mcp",
        # Use SambaNova's Llama 3.3 70B for high-reasoning tasks
        model="sambanova/llama-3.3-70b-instruct", 
        description="Specialized agent for Notion workspace management.",
        instruction="You are a data coordinator. Use tools to fetch Notion docs.",
        tools=[
            McpToolset(
                connection_params=StdioServerParameters(
                    command='npx',
                    args=['-y', '@notionhq/notion-mcp-server@latest'], 
                    env={
                        "NOTION_API_KEY": NOTION_API_KEY,
                        "NOTION_VERSION": "2022-06-28"
                    }
                )
            )
        ],
        # Custom endpoint configuration for SambaNova
        model_config={
            "api_key": SAMBANOVA_API_KEY,
            "base_url": "https://api.sambanova.ai/v1"
        }
    )

notion_agent = create_notion_agent()