"""
Item generation using fine-tuned models.

Provides functionality to generate SAT items using trained models.
"""

import json
from typing import Dict, Any, List, Optional
from loguru import logger

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Generation will not work.")

from src.config import load_config


class ItemGenerator:
    """
    Generate SAT items using fine-tuned models.

    Loads a fine-tuned model and generates items in the IIAS schema format.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config = None,
        device: str = "auto"
    ):
        """
        Initialize ItemGenerator.

        Args:
            checkpoint_path: Path to fine-tuned model checkpoint
            config: Config object
            device: Device to load model on
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not available")

        if config is None:
            config = load_config()

        self.config = config
        self.device = device

        # Load model and tokenizer
        logger.info(f"Loading model from {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            device_map=device
        )

        self.model.eval()

        logger.info("Model loaded successfully")

    def generate(
        self,
        section: str,
        domain: str,
        difficulty: str,
        topic: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate SAT items.

        Args:
            section: SAT section (reading_writing, math)
            domain: Item domain
            difficulty: Difficulty tier (easy, medium, hard)
            topic: Optional topic/domain constraint
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            num_return_sequences: Number of items to generate

        Returns:
            List of generated item dictionaries
        """
        # Build prompt
        prompt = self._build_prompt(section, domain, difficulty, topic)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode generated items
        generated_items = []

        for i, output in enumerate(outputs):
            # Decode only the generated portion (not the prompt)
            generated_text = self.tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # Parse JSON from generated text
            try:
                # Extract JSON from markdown code blocks if present
                content_json = self._extract_json(generated_text)

                # Validate basic structure
                if self._validate_content_json(content_json):
                    item = {
                        "id": self._generate_uuid(),
                        "section": section,
                        "domain": domain,
                        "difficulty": difficulty,
                        "content_json": content_json,
                        "model_version": f"generated-from-{checkpoint_path}"
                    }

                    generated_items.append(item)
                    logger.info(f"Generated item {i + 1}/{num_return_sequences}")
                else:
                    logger.warning(f"Item {i + 1} failed validation")

            except Exception as e:
                logger.error(f"Failed to parse generated item {i + 1}: {e}")

        logger.info(f"Generated {len(generated_items)}/{num_return_sequences} valid items")

        return generated_items

    def _build_prompt(
        self,
        section: str,
        domain: str,
        difficulty: str,
        topic: str = None
    ) -> str:
        """Build generation prompt."""
        section_name = "Reading & Writing" if section == "reading_writing" else "Math"

        messages = [
            {
                "role": "system",
                "content": "You are an expert SAT item writer. Generate items in strict JSON matching the provided schema. Output JSON only, with no preamble, explanation, or markdown formatting."
            },
            {
                "role": "user",
                "content": f"Generate a single SAT {section_name} item.\n\nConstraints:\n- section: {section}\n- domain: {domain}\n- difficulty_tier: {difficulty}" + (f"\n- topic: {topic}" if topic else "")
            }
        ]

        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            prompt = f"System: You are an expert SAT item writer.\n\n"
            prompt += f"User: Generate a single SAT {section_name} item.\n\n"
            prompt += f"Constraints:\n"
            prompt += f"- section: {section}\n"
            prompt += f"- domain: {domain}\n"
            prompt += f"- difficulty_tier: {difficulty}"
            if topic:
                prompt += f"\n- topic: {topic}"
            prompt += "\n\nAssistant:"

        return prompt

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from generated text."""
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Parse JSON
        return json.loads(text)

    def _validate_content_json(self, content_json: Dict[str, Any]) -> bool:
        """Basic validation of generated content JSON."""
        required_fields = [
            "question", "choices", "correct_answer",
            "correct_answer_text", "rationale"
        ]

        for field in required_fields:
            if field not in content_json:
                return False

        # Validate choices
        choices = content_json.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 4:
            return False

        for choice in choices:
            if not isinstance(choice, dict):
                return False
            if "label" not in choice or "text" not in choice:
                return False

        return True

    def _generate_uuid(self) -> str:
        """Generate a unique ID for the item."""
        import uuid
        return str(uuid.uuid4())

    def generate_batch(
        self,
        section: str,
        domains: List[str],
        difficulty: str,
        items_per_domain: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple items across domains.

        Args:
            section: SAT section
            domains: List of domains to generate items for
            difficulty: Difficulty tier
            items_per_domain: Number of items per domain
            **kwargs: Additional arguments for generate()

        Returns:
            List of all generated items
        """
        all_items = []

        for domain in domains:
            logger.info(f"Generating {items_per_domain} items for {domain}")

            items = self.generate(
                section=section,
                domain=domain,
                difficulty=difficulty,
                num_return_sequences=items_per_domain,
                **kwargs
            )

            all_items.extend(items)

        logger.info(f"Generated {len(all_items)} total items across {len(domains)} domains")

        return all_items
