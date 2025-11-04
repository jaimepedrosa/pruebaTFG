from smolagents import CodeAgent, Tool, TransformersModel
from transformers import BitsAndBytesConfig
import torch

# Define the financial calculator tool
class FinancialCalculatorTool(Tool):
    name = "financial_calculator"
    description = """Computes compound interest using the formula A = P * (1 + r/n)^(n*t) where:
    - P is the principal amount
    - r is the annual interest rate as a decimal (e.g., 0.06 for 6%)
    - n is the number of times interest is compounded per year
    - t is the time in years"""

    inputs = {
        "P": {"type": "number", "description": "Principal amount"},
        "r": {"type": "number", "description": "Annual interest rate as decimal"},
        "n": {"type": "integer", "description": "Times compounded per year"},
        "t": {"type": "number", "description": "Time in years"}
    }
    output_type = "number"

    def forward(self, P: float, r: float, n: int, t: float) -> float:
        """Calculate compound interest."""
        return P * (1 + r/n) ** (n*t)

# Configure 8-bit quantization
model_name = "SUFE-AIFLM-Lab/Fin-R1"

bnb_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# Load model
smolagents_model = TransformersModel(
    model_id=model_name,
    device_map="auto",
    model_kwargs={
        "quantization_config": bnb_8bit,  # Only quantization in model_kwargs
    }
)

# Initialize the agent
agent = CodeAgent(
    tools=[FinancialCalculatorTool()],
    model=smolagents_model
)

# Run the agent
user_query = """I have $10,000 to invest. The bank offers an interest rate of 6% per year,
compounded monthly (12 times per year). How much money will I have after 10 years?"""

result = agent.run(user_query)
print(result)