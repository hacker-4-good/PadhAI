from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction 
import dspy 
from dotenv import load_dotenv
from dspy.retrieve.chromadb_rm import ChromadbRM
from pydantic import BaseModel, Field
load_dotenv()

llm = dspy.PremAI(project_id=8883, api_key='XPa8IkQXNLt5zPbVS4CI7Pfot3H2FPFIV3', temperature=1)
rm = ChromadbRM(
    collection_name = "Algorithm", 
    persist_directory = "DB", 
    embedding_function = SentenceTransformerEmbeddingFunction()
)

dspy.configure(lm = llm, rm=rm)

class QuerySignature(dspy.Signature):
    '''
    Provide complete and to-the-point answers to student queries regarding their subjects, including both theoretical questions and numerical problems, using content from textbooks.
    *You are great in mathematics so show proper steps to solve numericals*
    '''
    context = dspy.InputField(desc="may contain relevant facts from textbooks")
    question: str = dspy.InputField(desc="Student's question, either theoretical or numerical")
    answer: str = dspy.OutputField(desc="Complete and to-the-point answer")

class QuizInput(BaseModel):
    topic: str = Field(description="The topic for the quiz")
    context: list[str] = Field(description="Relevant context from ChromaDB")

class QuizOption(BaseModel):
    option: str = Field(description="A possible answer option")

class QuizOutput(BaseModel):
    question: str = Field(description="The generated quiz question")
    options: list[QuizOption] = Field(description="The list of answer options")
    correct_option: int = Field(ge=0, le=3, description="The index of the correct answer option")

class QuizSignature(dspy.Signature):
    """Generate a quiz question on a user-provided topic with 4 answer options."""
    input: QuizInput = dspy.InputField()
    output: QuizOutput = dspy.OutputField()


class ChatbotRAG(dspy.Module):
    def __init__(self, num_passage=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passage)
        self.generate_answer = dspy.ChainOfThought(signature=QuerySignature)

    def forward(self, question):
        context = self.retrieve(question).passages
        print(context)
        prediction = self.generate_answer(context = context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

class QuizRAG(dspy.Module):
    def __init__(self):
        super().__init__() 
        self.retrieve = dspy.Retrieve() 
        self.generate_quiz = dspy.TypedPredictor(QuizSignature)
    def forward(self, quiz_text):
        context = self.retrieve(quiz_text).passages 
        quiz_input = QuizInput(topic=quiz_text, context=context)
        prediction = self.generate_quiz(input=quiz_input)
        return prediction