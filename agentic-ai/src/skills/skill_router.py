"""
Placeholder for SkillRouter
"""
import logging
from typing import Dict, Any, List
from .base_skill import SkillRegistry, SkillType, SkillConfiguration

logger = logging.getLogger(__name__)

class SkillRouter:
    """
    A component that intelligently selects and sequences skills based on the task description,
    context, and available skills in the global registry.
    """
    def __init__(self):
        self.skill_registry = SkillRegistry()
        logger.info("Initialized SkillRouter with global SkillRegistry.")

    def create_plan(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a plan by intelligently selecting the most appropriate skill(s)
        based on the task description and available skills.
        """
        logger.info(f"SkillRouter: Creating plan for task: '{task}'.")
        task_lower = task.lower()
        
        # Candidate skills and their relevance scores
        candidate_skills = []
        
        for skill_name, skill_instance in self.skill_registry._skills.items():
            skill_config: SkillConfiguration = skill_instance.config
            score = 0
            
            # Score based on keyword matching in name and description
            skill_name_lower_sanitized = skill_config.name.lower().replace("_", " ")
            skill_description_lower = skill_config.description.lower()
            
            if skill_name_lower_sanitized in task_lower:
                score += 3
            
            for keyword in task_lower.split():
                if keyword in skill_description_lower:
                    score += 1
            
            # Score based on skill type (can be refined further)
            # Example: if task explicitly mentions 'retrieve data', data retrieval skills get a boost
            if "retrieve data" in task_lower and skill_config.skill_type == SkillType.DATA_RETRIEVAL:
                score += 2
            if "analyze" in task_lower and skill_config.skill_type == SkillType.ANALYSIS:
                score += 2
            
            # Score based on domain context
            if skill_config.domain_context and skill_config.domain_context.lower() in task_lower:
                score += 2
                
            if score > 0:
                candidate_skills.append((score, skill_instance))
        
        # Sort candidates by score in descending order
        candidate_skills.sort(key=lambda x: x[0], reverse=True)
        
        if candidate_skills:
            best_skill = candidate_skills[0][1]
            logger.info(f"SkillRouter: Found best matching skill '{best_skill.config.name}' with score {candidate_skills[0][0]}.")
            
            # For now, return a plan with the single best skill.
            # Future enhancements could involve sequencing multiple skills.
            return [
                {
                    "skill_name": best_skill.config.name,
                    "input": {
                        "task_description": task,
                        "context": context
                    }
                }
            ]
        
        logger.warning(f"SkillRouter: No relevant skill found for task: '{task}'. Returning empty plan.")
        return []