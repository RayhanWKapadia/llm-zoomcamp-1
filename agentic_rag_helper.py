INSTRUCTIONS = '''
Your task is to answer questions from a course about large-language models and RAG.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context.
respond with "I don't know."
'''

from toyaikit.tools import Tools
from toyaikit.llm import OpenAIClient
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner
from openai import OpenAI

class AgenticRAGBase:

    def __init__(self, index, client, instructions=INSTRUCTIONS, model='gpt-5.4-mini'):
        self.tools = Tools()
        self.client = client
        self.response = None
        self.index = index
        self.model = model
        self.course = "llm-zoomcamp"
        self.instructions = instructions
        self.loop_count = 0

    def agenticRAG(self, question: str):    
        openai_client = OpenAIClient(
            model=self.model,
            client=OpenAI()
        )
        self.tools.add_tool(self.agentic_search)

        runner = OpenAIResponsesRunner(
            tools=self.tools,
            developer_prompt=self.instructions,
            llm_client=openai_client
        )
        result = runner.loop(prompt=question)

        print(self.loop_count)

        return result.last_message

    def agentic_search(self, query: str) -> list[dict]:      
        """
        You're a course teaching assistant. Answer the student's question using the search tool. 
        Make multiple searches with different keywords before answering."
        """
        self.loop_count+=1
        return self.index.search(
            query,
            num_results=5,
            boost_dict={"content": 2.0},
        )
