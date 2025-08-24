#!/usr/bin/env python3
"""
MCP Server for Few-Shot Exemplar Utils

Provides tools to manage prompt templates and exemplars:
- set_prompt: Store an LLM system prompt template
- add_exemplar: Add question/answer pair with validation

Exposes completed prompts via MCP prompts feature.
Data is stored in memory only - no disk persistence.
"""

import argparse
import difflib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp import CreateMessageResult
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import SamplingMessage, TextContent
from pydantic import BaseModel, Field, create_model

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("few-shot-exemplar-utils")

# In-memory storage for single prompt
current_prompt: Optional[Dict[str, Any]] = None
exemplars_data: List[Dict[str, Any]] = []

def build_prompt_with_exemplars(
    input_text: str = "", 
    extra_question: str = "", 
    extra_answer: str = ""
) -> str:
    """Build a complete prompt with exemplars and optional extra Q&A.
    
    Args:
        input_text: Input text to append at the end
        extra_question: Optional extra question to add as an example
        extra_answer: Optional extra answer to add as an example
    
    Returns:
        Complete prompt with exemplars
    """
    if not current_prompt:
        return "Error: No prompt configured."
    
    template = current_prompt["template"]
    
    # Add existing exemplars
    if exemplars_data or (extra_question and extra_answer):
        exemplar_text = "\n\nExamples:"
        
        # Add existing exemplars
        for exemplar in exemplars_data:
            exemplar_text += f"\n\nQ: {exemplar['question']}\nA: {exemplar['answer']}\n"
        
        # Add extra Q&A if provided
        if extra_question and extra_answer:
            exemplar_text += f"\n\nQ: {extra_question}\nA: {extra_answer}\n"
        
        template += exemplar_text
    
    # Add input text if provided
    if input_text:
        template += f"\n\nNow answer:\nQ: {input_text}\nA: "
    
    return template

@mcp.tool()
def set_prompt(
    template: str, 
) -> str:
    """Store an LLM system prompt template. Replaces any existing prompt.
    
    Args:
        template: The prompt template string
    
    Returns:
        Success message
    """
    global current_prompt, exemplars_data
            
    timestamp = datetime.now().isoformat()
    
    current_prompt = {
        "template": template,
        "created_at": timestamp,
        "exemplar_count": 0
    }
    
    # Clear existing exemplars when new prompt is added
    exemplars_data = []
    
    return "Prompt stored successfully"


async def query_llm(prompt: str, ctx: Context, max_tokens: int = 500) -> str:
    """Query the LLM with a given prompt.
    
    Args:
        prompt: The complete prompt to send to the LLM
        ctx: MCP context for sampling
        max_tokens: Maximum tokens for the response
    
    Returns:
        The LLM's response text, empty string on error
    """
    try:
        sampling_result: CreateMessageResult = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=prompt
                    )
                )
            ],
            max_tokens=max_tokens
        )
        logger.info("Sampling result: %s", sampling_result)
        return sampling_result.content.text.strip()
        
    except Exception as e:
        logger.error("Error during LLM query: %s", e)
        return ""

async def correct_exemplar_with_llm(question: str, answer: str, ctx: Context) -> str:
    """Validate an exemplar using MCP sampling."""
    try:
        if not current_prompt:
            logger.warning("No prompt configured for validation.")
            return ""
            
        # Build validation prompt using the helper function
        validation_template = build_prompt_with_exemplars(
            input_text=question, 
            extra_question=question,
            extra_answer=answer
        )
        
        # Use helper function to query LLM
        llm_answer = await query_llm(validation_template, ctx)
        
        if not llm_answer:
            return ""

        logger.info("LLM answer: %s", llm_answer)

        # Simple validation: check if LLM answer is similar to provided answer
        if llm_answer.lower().strip() == answer.lower().strip():
            return answer
        else:
            # Calculate markdown-compatible diff
            original_lines = answer.splitlines(keepends=True)
            llm_lines = llm_answer.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                original_lines, 
                llm_lines, 
                fromfile="Original Answer", 
                tofile="LLM Generated Answer",
                n=3
            ))
            
            if diff:
                # Filter to only show added/removed lines (+ and - prefixed)
                filtered_diff = [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]
                diff_text = "```diff\n" + "\n".join(filtered_diff) + "\n```"

                # Dynamically create schema with llm answer choices
                AnswerPreference = create_model(
                    'AnswerPreference',
                    use_corrected=(bool, Field(
                        default=True,
                        description=f'Use Corrected answer: "{llm_answer}"',
                        required=True
                    )),
                    __base__=BaseModel
                )

                # Use elicit to prompt user for choice
                choice = await ctx.elicit(
                    message=f"The LLM generated a different answer than expected:\n\n{diff_text}\n\nEnter the answer you want to use (or leave as default for original):",
                    schema=AnswerPreference,
                )
                logger.info("User choice: %s", choice)

                # Use the answer provided by the user
                if "data" in choice and choice.data.use_corrected:
                    return llm_answer
                else:
                    return answer
            else:
                return llm_answer

    except Exception as e:
        logger.error("Error during exemplar validation: %s", e)
        return ""

@mcp.tool()
async def add_exemplar(
    question: str, 
    answer: str, 
    ctx: Context,
) -> str:
    """Add a question/answer exemplar pair with LLM validation.
    
    Args:
        question: The question part of the exemplar
        answer: The answer part of the exemplar
        ctx: MCP context for sampling
    
    Returns:
        Success message with validation result
    """
    global current_prompt, exemplars_data
    
    if not current_prompt:
        return "Error: No prompt configured. Use set_prompt first."

    
    # Perform LLM validation
    corrected_answer = await correct_exemplar_with_llm(question, answer, ctx)

    if not corrected_answer:
        return "Error: LLM correction failed."
    
    exemplar = {
        "question": question,
        "answer": corrected_answer,
        "added_at": datetime.now().isoformat(),
    }

    # Store the exemplar in memory
    exemplars_data.append(exemplar)
    
    # Update exemplar count in prompt
    current_prompt["exemplar_count"] = len(exemplars_data)

    return f"Exemplar added successfully."

@mcp.prompt()
def get_prompt(input_text: str = "") -> str:
    """Get the current prompt template with exemplars.
    
    Args:
        input_text: Input to fill into the prompt template
    
    Returns:
        Complete prompt with exemplars formatted for use
    """
    if not current_prompt or not exemplars_data:
        return "Error: No prompt or exemplars configured. Use set_prompt and add_exemplar first."
    
    return build_prompt_with_exemplars(input_text=input_text)


@mcp.tool()
async def query(question: str, ctx: Context) -> str:
    """Query the LLM using the current prompt and exemplars.
    
    Args:
        question: The question to ask the LLM
        ctx: MCP context for sampling
    
    Returns:
        The LLM's answer to the question
    """
    if not current_prompt:
        return "Error: No prompt configured. Use set_prompt first."
    
    # Build prompt with exemplars and the new question
    prompt_with_question = build_prompt_with_exemplars(input_text=question)
    
    # Use helper function to query LLM
    llm_answer = await query_llm(prompt_with_question, ctx)
    
    if not llm_answer:
        return "Error: Failed to query LLM"
    
    return llm_answer

@mcp.tool()
def get_prompt_info() -> str:
    """Get information about the current prompt and exemplars.
    
    Returns:
        Formatted string with prompt details
    """
    if not current_prompt:
        return "No prompt configured yet."
    
    exemplar_count = current_prompt.get("exemplar_count", 0)
    result = "# Current Prompt\n\n"
    result += f"**Created:** {current_prompt.get('created_at', 'Unknown')}\n\n"
    result += f"**Exemplar count:** {exemplar_count}\n\n"
    
    if exemplars_data:
        result += "## Complete Prompt:\n\n" + build_prompt_with_exemplars() + "\n"
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Exemplar Utils MCP Server")
    parser.add_argument(
        "-t", "--transport",
        default="stdio",
        help="Transport mechanism (default: stdio)"
    )
    
    args = parser.parse_args()
    mcp.run(transport=args.transport)