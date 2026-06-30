from dotenv import load_dotenv
from langsmith import traceable
import inspect
from react_prompt import ReactPrompt
from ollama import chat
import re
load_dotenv()
MAX_ITERATIONS=10
MODEL="qwen3:1.7b"
#  provided by Olamma , local

#    Tools
def get_metadata(func):
    func = inspect.unwrap(func)
    res = f"{func.__name__}{inspect.signature(func)} -  {inspect.getdoc(func)}"
    return res

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
def apply_discount(price: str, discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    price = float(price)
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


#  Agent Loop

tool_dict= {'get_product_price':get_product_price,'apply_discount':apply_discount}
print(type(tool_dict.keys()))
tool_names = ", ".join(tool_dict.keys())


@traceable(run_type="llm",name="Ollama Chat") 
def ollama_chat_traced(messages,options):
    return chat(model=MODEL, messages=messages,options=options)

def getToolDescription(tools_dict:dict):
    result=[]
    for key,val in tools_dict.items():
        meteData=get_metadata(val)
        result.append(meteData)
    return "\n".join(result)



@traceable(name="Olamma Agent Loop")
def run_agent(question:str):
    print(f"Question {question}")
    print("--" * 60)
    react_prmpt = ReactPrompt.getReactPrompt(getToolDescription(tool_dict),tool_names)
    prompt  = react_prmpt.format(question=question)
    scratchpad=""


    # Agent loop
    finalAnswer=None
    i=1
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"--" * 60)
        print(f"Iteration {i}")
        i+=1
        full_prompt = prompt+scratchpad

        response = ollama_chat_traced(
            messages=[{'role': 'user', 'content': f'{full_prompt}'}]
            , options={"temperature": 0,"stop": ["\nObservation"]}
            )
        output=response.message.content

        print(f"LLM output: {output}")

        match = re.search(r"Final Answer:\s*(.*)", output, re.DOTALL)
        if(match):
            finalAnswer=match.group(1).strip()
            print(f"completed Generating Final Answer {finalAnswer}")
            return finalAnswer

        else:
            action_match = re.search(r"Action:\s*(\S+)", output)
            action_name = action_match.group(1) if action_match else None

   


            input_match = re.search(r"Action Input:\s*(.*)", output)
            action_input = input_match.group(1) if input_match else None

            print(f"Tool Selected: {action_name}")
            print(f"Args: {action_input}")
            raw_args = [x.strip() for x in action_input.split(",")]
            args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

            toolNameToInvoke=action_name
            toolToInvoke=tool_dict[toolNameToInvoke]
            # Invoking the tool

            if action_name not in tool_names:
                observation= f"Error: Tool '{action_name}' not found. Available tools: {tool_names}"
            else:
                observation = str(toolToInvoke(*args))

            print(f"Tool Result {observation}")

            scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: max Iteration reached without Answer")
    return None        
  



if __name__ == "__main__":
    print('Hello langChain Agent')
    result= run_agent('What is the price of a laptop after applying gold discount?')
