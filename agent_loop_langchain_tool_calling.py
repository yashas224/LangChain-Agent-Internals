from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable

load_dotenv()
MAX_ITERATIONS=10
MODEL="qwen3:1.7b"
#  provided by Olamma , local


#   LangChain Tools

@tool
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

@tool
def apply_discount(price: float, discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""

    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

#  Agent Loop

@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tool_dict= {t.name:t for t in tools}
    print(f"tool Dict {tool_dict}")

    llm  = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools=tools)

    print(f"Question {question}")
    print("--" * 60)

    conversation = [
        SystemMessage(content=[
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
                "ask them politely which tier to use — do NOT assume one."
        ]),
        HumanMessage(content=question)
    ]

    # Agent loop
    finalAnswer=None
    i=1
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"--" * 60)
        print(f"Iteration {i}")
        i+=1

        response = llm_with_tools.invoke(conversation)
        conversation.append(response)

        if not response.tool_calls:
            print(f"completed Generating Final Answer {response.content}")
            finalAnswer=response.content
            return finalAnswer
        else:
            for tool_call in response.tool_calls:
                # View tool calls made by the model
                print(f"Tool Selected: {tool_call['name']}")
                print(f"Args: {tool_call['args']}")

                toolNameToInvoke=tool_call['name']
                toolToInvoke=tool_dict[toolNameToInvoke]

                # Invoking the tool
                observation = toolToInvoke.invoke(tool_call)
                print(f"Tool Result {observation}")
                conversation.append(observation)


    print("ERROR: max Iteration reached without Answer")
    return None        
  



if __name__ == "__main__":
    print('Hello langChain Agent')
    result= run_agent('What is the price of a laptop after applying gold discount?')
