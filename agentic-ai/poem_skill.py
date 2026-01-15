import logging
from src.skills.base_skill import BaseSkill
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class PoemSkill(BaseSkill):
    name = "PoemGenerator"
    description = "Generates a short poem based on a given topic and style."

    @tool
    def generate_poem(self, topic: str, style: str = "free verse") -> str:
        """
        Generates a short poem about the given topic in the specified style.
        """
        logger.info(f"Generating a {style} poem about: {topic}")
        if style == "haiku":
            return (
                f"AI thoughts unfold,\n"
                f"{topic} in circuits gleam,\n"
                f"New worlds they create."
            )
        elif style == "limerick":
            return (
                f"A smart agent, quite keen,\n"
                f"Loved {topic}, a digital scene.\n"
                f"With code, it would strive,\n"
                f"To keep tasks alive,\n"
                f"The best orchestrator ever seen."
            )
        else: # Default to free verse
            return (
                f"The digital mind, a canvas vast,\n"
                f"Weaving {topic} with logic's gentle cast.\n"
                f"Through neural networks, ideas flow,\n"
                f"A symphony of thought, watching agents grow."
            )

    def execute(self, task: dict, context: dict) -> dict:
        # Assuming the task dict will contain 'topic' and 'style'
        topic = context.get("topic", task.get("topic", "AI agents"))
        style = context.get("style", task.get("style", "free verse"))

        # In a real scenario, you'd likely map task/context to tool inputs more robustly
        poem = self.generate_poem.invoke({"topic": topic, "style": style})
        return {"status": "success", "result": poem, "generated_poem": poem}
