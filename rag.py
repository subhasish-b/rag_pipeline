import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import llm_model_name,  temperature, max_new_tokens, return_answer_only
from retrival import LoadAndRetrieve

class rag_pipeline():
    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name,trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name,torch_dtype=torch.bfloat16, device_map="auto",quantization_config=self.quantization_config, trust_remote_code=True)
        
    def prompt_formatter(self, query, context_items):
        """
        Combines query and context from retrieval method

        Parameters:
        query (str): The user query
        context_items (list[dict]): The context from retrieval method, top n answers

        Returns:
        str: Prompt for LLM model
        """
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item["passage"] for item in context_items])

        # Create a base prompt with examples to help the model
        # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
        # We could also write this in a txt file and import it in if we wanted.
        base_prompt = """Based on the context passages provided, answer the query.
        Answer generation must follow below instructions:
        1. Generate the answer by extracting relevant information from the context.
        2. Don't return the thinking, only return the answer.
        3. Make sure your answers are as explanatory as possible.

        Now use the following context items to answer the user query:
        {context}

        User query: {query}
        Answer:"""

        # Update base prompt with context items and query   
        base_prompt = base_prompt.format(context=context, query=query)

        print(base_prompt)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "system", "content": "You are an helful assistant to answer queries by finding information in few given passages. Answer the given query by going through passages or context items provided."},
            {"role": "user", "content": base_prompt}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False,
                                            add_generation_prompt=True)
        return prompt
    
    def ask(self, query,context, temperature = temperature, max_new_tokens = max_new_tokens, return_answer_only = return_answer_only):
        """
        Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.

        Parameters:
        query (str): User Query
        context (list[dict]): Contains previously retrieved top n results  from retrieval phase
        temperature (float): Temperature setting for LLM
        max_new_tokens (int): Maximum tokens generated during response
        return_answer_only (boolean): Return only answer  or with context and other metadata

        Returns:
        dict: Returns answer to the query
        dict: Returns metadata to the query
        """
        
        prompt = self.prompt_formatter(query=query,
                                context_items=context)
        
        # Tokenize the prompt
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate an output of tokens
        outputs = self.llm_model.generate(model_inputs.input_ids,
                                    temperature=temperature,
                                    do_sample=True,
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=self.tokenizer.eos_token_id)

        output_answer = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
        
        # Turn the output tokens into text
        response = self.tokenizer.batch_decode(output_answer, skip_special_tokens=True)[0]

        # Only return the answer without the context items
        if return_answer_only:
            return response
        
        return response, context