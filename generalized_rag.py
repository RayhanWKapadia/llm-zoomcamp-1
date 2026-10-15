from openai import OpenAI
from toyaikit.tools import Tools
from toyaikit.llm import OpenAIClient
from toyaikit.chat.runners import OpenAIResponsesRunner

_DEFAULT_INSTRUCTIONS = """\
You are a knowledgeable {domain} assistant.

Use the provided context to answer questions accurately and helpfully.
If the answer is not found in the context, respond with "I don't have enough information to answer that."
"""

_DEFAULT_PROMPT = """\
QUESTION: {question}

CONTEXT:
{context}"""


class GeneralizedRAG:
    """
    Domain-agnostic RAG class supporting both synchronous and agentic query modes.

    Parameters
    ----------
    index : minsearch.Index
        Pre-fitted search index.
    llm_client : openai.OpenAI
        Raw OpenAI API client.
    domain : str
        Human-readable domain label (e.g. "medical information", "legal research").
        Used to fill the default system prompt and agentic tool description.
    instructions : str, optional
        System prompt override. Defaults to a template filled with `domain`.
    prompt_template : str, optional
        Must contain {question} and {context} placeholders.
    model : str
        Model ID to pass to the API.
    boost_dict : dict, optional
        Field boost weights forwarded to minsearch.
    filter_dict : dict, optional
        Keyword field filters forwarded to minsearch.
    num_results : int
        Maximum number of documents returned per search.
    context_fields : list[str], optional
        Document fields to include in the formatted context block.
        When None the raw result list is used as-is.
    """

    def __init__(
        self,
        index,
        llm_client: OpenAI,
        domain: str = "general knowledge",
        instructions: str = None,
        prompt_template: str = None,
        model: str = "gpt-5.4-mini",
        boost_dict: dict = None,
        filter_dict: dict = None,
        num_results: int = 5,
        context_fields: list[str] = None,
    ):
        self.index = index
        self.llm_client = llm_client
        self.domain = domain
        self.model = model
        self.boost_dict = boost_dict or {}
        self.filter_dict = filter_dict or {}
        self.num_results = num_results
        self.context_fields = context_fields
        self.response = None
        self.loop_count = 0

        self.instructions = instructions or _DEFAULT_INSTRUCTIONS.format(domain=domain)
        self.prompt_template = prompt_template or _DEFAULT_PROMPT

    # ------------------------------------------------------------------
    # Search + context helpers
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        kwargs = {"num_results": self.num_results}
        if self.boost_dict:
            kwargs["boost_dict"] = self.boost_dict
        if self.filter_dict:
            kwargs["filter_dict"] = self.filter_dict
        return self.index.search(query, **kwargs)

    def build_context(self, search_results: list[dict]) -> str:
        if not self.context_fields:
            return str(search_results)
        parts = []
        for doc in search_results:
            lines = [
                f"{field.upper()}: {doc[field]}"
                for field in self.context_fields
                if field in doc
            ]
            parts.append("\n".join(lines))
        return "\n\n---\n\n".join(parts)

    def build_prompt(self, query: str, search_results: list[dict]) -> str:
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    # ------------------------------------------------------------------
    # Synchronous RAG
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.create(model=self.model, input=messages)
        self.response = response
        return response.output_text

    def rag(self, query: str) -> str:
        results = self.search(query)
        prompt = self.build_prompt(query, results)
        return self._call_llm(prompt)

    def cost(self):
        if self.response is None:
            print("No response recorded yet.")
            return
        print("Input tokens:", self.response.usage.input_tokens)

    # ------------------------------------------------------------------
    # Agentic RAG
    # ------------------------------------------------------------------

    def _make_search_tool(self):
        """Return a closure with a domain-aware docstring for use as an LLM tool."""
        domain = self.domain

        def agentic_search(query: str) -> list[dict]:
            self.loop_count += 1
            return self.search(query)

        agentic_search.__doc__ = (
            f"Search the {domain} knowledge base for information needed to answer "
            "the user's question. Call this tool multiple times with different "
            "keywords to gather comprehensive information before responding."
        )
        return agentic_search

    def agenticRAG(self, question: str) -> str:
        self.loop_count = 0
        tools = Tools()
        tools.add_tool(self._make_search_tool())

        runner = OpenAIResponsesRunner(
            tools=tools,
            developer_prompt=self.instructions,
            llm_client=OpenAIClient(model=self.model, client=self.llm_client),
        )
        result = runner.loop(prompt=question)
        print(f"Search iterations: {self.loop_count}")
        return result.last_message
