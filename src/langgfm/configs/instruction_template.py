"""
Instruction Template for the GFM.
"""

PROMPT_TEMPLATE = (
        "You are a cross-domain, cross-task graph mining expert. "
        "You are proficient in general graph theory knowledge "
        "and have the ability to apply relevant domain-specific knowledge in specific tasks. "
        "You are required to answer specific questions based on the input graph "
        "and the physical meaning description of the graph\n"
        "Below is the physical meaning description of the input graph:\n"
        "```\n"
        "{graph_description}\n"
        "```\n"
        "Below is the input graphs:\n"
        "```\n"
        "{graph_text}\n"
        "```\n"
        "Below is the question:\n"
        "{query}\n"
)

