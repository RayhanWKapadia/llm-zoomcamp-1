INSTRUCTIONS = '''
Your task is to answer questions from a course about large-language models and RAG.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context.
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.response = None

    def search(self, query, num_results=5):
        boost_dict = {'content': 2.0}
        # filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict
            # filter_dict=filter_dict
        )

    def build_context(self, search_results):
        return search_results

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        
        self.response = response

        return response.output_text

    def cost(self):
        print("Token cost: ", self.response.usage.input_tokens)

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

    def agentic_search(self, query, num_results=5):
        boost_dict = {'content': 2.0}
        # filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict
            # filter_dict=filter_dict
        )