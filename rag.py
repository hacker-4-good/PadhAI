import json
from langchain_mistralai import ChatMistralAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import dspy 
from knowledge_base import qdrant
import os
from config import MISTRAL_API_KEY, QDRANT_API_KEY

llm = dspy.LM(
    model = "mistral-medium", 
    api_key=MISTRAL_API_KEY, 
    api_base="https://api.mistral.ai/v1/"
)

dspy.settings.configure(lm = llm)

class QuerySignature(dspy.Signature):
    '''
    Provide complete and to-the-point answers to student queries regarding their subjects, including both theoretical questions and numerical problems, using content from textbooks.
    *You are great in mathematics so show proper steps to solve numericals*
    '''
    context = dspy.InputField(desc="may contain relevant facts from textbooks")
    question: str = dspy.InputField(desc="Student's question, either theoretical or numerical")
    answer: str = dspy.OutputField(desc="Complete and to-the-point answer")


class ChatbotRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(signature=QuerySignature)

    def forward(self, question):
        context = qdrant.search(
            query=question,
            search_type="similarity_score_threshold"  
        )
        prediction = self.generate_answer(context = context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

class QuizRAG:

    def __init__(self):

        # ---------------------------
        # QDRANT CLOUD (FIXED)
        # ---------------------------
        self.client = QdrantClient(
            url="https://6d63122d-7b93-4b00-81e3-b454b3363930.eu-west-2-0.aws.cloud.qdrant.io",
            api_key=QDRANT_API_KEY
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="Content",
            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.6}
        )

        # ---------------------------
        # LLM (STABLE JSON)
        # ---------------------------
        self.llm = ChatMistralAI(
            model="mistral-medium",
            api_key=MISTRAL_API_KEY,
            base_url="https://api.mistral.ai/v1/",
            temperature=0
        )

        # ---------------------------
        # PROMPT (STRICT JSON)
        # ---------------------------
        self.prompt = PromptTemplate(
            input_variables=["topic", "context"],
            template="""
            You are a quiz generation system.

            STRICT RULES:
            - Output MUST be valid JSON
            - NO explanation
            - NO markdown
            - ONLY JSON

            Schema:
            {{
            "topic": "<string>",
            "questions": [
                {{
                "question": "<string>",
                "options": ["A", "B", "C", "D"],
                "answer": "<correct option>"
                }}
            ]
            }}

            Rules:
            - Generate exactly 5 questions
            - Each must have 4 options
            - Answer must match one option exactly
            - Use ONLY the given context

            Topic: {topic}

            Context:
            {context}
            """
        )

        # ---------------------------
        # CHAIN
        # ---------------------------
        self.chain = (
            {
                "context": self.retriever | self.format_docs,
                "topic": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # ---------------------------
    # FORMAT DOCS
    # ---------------------------
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------------------------
    # SAFE JSON PARSER (IMPORTANT)
    # ---------------------------
    def json_parser(self, text):
        try:
            return json.loads(text)
        except Exception:
            # 🔥 Fix common LLM issues
            text = text.strip()

            # remove markdown if present
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0]

            return json.loads(text)

    # ---------------------------
    # RETRY LOGIC (VERY IMPORTANT)
    # ---------------------------
    def generate(self, topic, retries=3):
        for _ in range(retries):
            try:
                raw_output = self.chain.invoke(topic)
                return self.json_parser(raw_output)
            except Exception:
                continue

        raise ValueError("Failed to generate valid JSON after retries")
