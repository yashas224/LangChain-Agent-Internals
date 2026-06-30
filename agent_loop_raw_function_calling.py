from dotenv import load_dotenv
from langsmith import traceable

from ollama import chat

load_dotenv()
MAX_ITERATIONS=10
MODEL="qwen3:1.7b"
#  provided by Olamma , local


#    Tools

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look for the price of the product in the catalog.

    Args:
        product: Name of the product to look for

    Returns:
        float: Proce of the product 
    """
    print(f'executing get_product_price for product {product}')
    prices = {'laptop':1299.99,'headphones':129.67,'keyboard':89.50}
    return prices.get(product,0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""

    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

#  Agent Loop

@traceable(run_type="llm",name="Ollama Chat") 
def ollama_chat_traced(messages):
    return chat(model=MODEL, messages=messages,tools=tools_for_llm)




@traceable(name="Olamma Agent Loop")
def run_agent(question:str):
    tool_dict= {'get_product_price':get_product_price,'apply_discount':apply_discount}
    print(f"tool Dict {tool_dict}")


    print(f"Question {question}")
    print("--" * 60)

    conversation = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]


    # Agent loop
    finalAnswer=None
    i=1
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"--" * 60)
        print(f"Iteration {i}")
        i+=1

        response = ollama_chat_traced(messages=conversation)
        ai_message=response.message
        conversation.append(ai_message)

        if not ai_message.tool_calls:
            finalAnswer=ai_message.content
            print(f"completed Generating Final Answer {finalAnswer}")

            return finalAnswer
        else:
            for call in ai_message.tool_calls:
                # View tool calls made by the model
                print(f"Tool Selected: {call.function.name}")
                print(f"Args: {call.function.arguments}")

                toolNameToInvoke=call.function.name
                toolToInvoke=tool_dict[toolNameToInvoke]

                # Invoking the tool
                observation = toolToInvoke(**call.function.arguments)
                print(f"Tool Result {observation}")
                conversation.append({'role': 'tool',  'tool_name': call.function.name, 'content': str(observation)})


    print("ERROR: max Iteration reached without Answer")
    return None        
  



if __name__ == "__main__":
    print('Hello langChain Agent')
    result= run_agent('What is the price of a laptop after applying gold discount?')
