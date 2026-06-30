class ReactPrompt:
    def getReactPrompt(tool_description,tool_names):
        react_prmpt = f"""
            STRICT RULES — you must follow these exactly
            1. NEVER guess or assume any product price.
            You MUST call get_product_price first to get the real price.
            2. Only call apply_discount AFTER you have received "
            a price from get_product_price. Pass the exact price "
            returned by get_product_price — do NOT pass a made-up number.
            3. NEVER calculate discounts yourself using math. 
            Always use the apply_discount tool.
            4. If the user does not specify a discount tier,
            ask them which tier to use — do NOT assume one.

            Answer the following questions as best you can. You have access to the following tools:

            {tool_description}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {{question}}
            Thought:""
            """
        return react_prmpt