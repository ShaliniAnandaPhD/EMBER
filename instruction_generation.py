from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from rl_framework import RLAgent

class InstructionGenerationAgent:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.rl_agent = RLAgent(self.model)

    def generate_instructions(self, optimized_routes, user_inputs):
        try:
            input_text = f"User inputs: {user_inputs}\nOptimized routes: {optimized_routes}\nInstructions:"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            output = self.model.generate(input_ids, max_length=200, num_return_sequences=1)
            instructions = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return instructions
        except Exception as e:
            logging.error(f"Error in InstructionGenerationAgent.generate_instructions: {str(e)}")
            return ""

    def process_feedback(self, feedback):
        try:
            # Process user feedback on the generated instructions for reinforcement learning
            reward = self.rl_agent.calculate_reward(feedback)
            self.rl_agent.update_model(reward)
        except Exception as e:
            logging.error(f"Error in InstructionGenerationAgent.process_feedback: {str(e)}")