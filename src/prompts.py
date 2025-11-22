class PromptManager:
    """Manages prompt templates for query expansion."""

    @staticmethod
    def get_prompt(query, type='gen'):
        if type == 'gen':
            # デフォルトでZero-shotを使用
            return PromptManager.get_strict_constraint_prompt(query)
        else:
            raise NotImplementedError(f"Template type {type} is not implemented.")

    @staticmethod
    def get_strict_constraint_prompt(query):
        """
        改良したプロンプト
        """
        system_instruction = (
            "You are an expert search assistant designed to provide precise, direct, and high-quality passages for information retrieval. "
            "Your goal is to write a passage that perfectly answers the user's query while strictly adhering to all constraints."
        )
        user_instruction = (
            f"Query: '{query}'\n\n"
            "Write a comprehensive and focused passage that directly answers this query. "
            "Follow these strict guidelines:\n"
            "1. **Specific Constraints**: Pay close attention to adjectives and modifiers (e.g., 'right' vs 'left', 'interior' vs 'exterior', specific numbers). Do not generalize.\n"
            "2. **Comparison**: If the query asks for a difference (e.g., 'vs', 'difference between'), explicitly contrast the entities rather than defining them separately.\n"
            "3. **Scope**: Stay strictly on topic. Do not include unrelated history, broad definitions, or tangential information unless necessary for the answer.\n"
            "4. **Density**: The passage should be dense with relevant keywords and facts matching the specific intent."
        )
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction}
        ]

    @staticmethod
    def get_legacy_zs_prompt(query):
        """旧プロンプト 比較用"""
        return [{
            "role": "system",
            "content": "You are PassageGenGPT, an AI capable of generating concise, informative, and clear pseudo passages on specific topics."
        }, {
            "role": "user",
            "content": f"Generate one passage that is relevant to the following query: '{query}'. The passage should be concise, informative, and clear"
        }]