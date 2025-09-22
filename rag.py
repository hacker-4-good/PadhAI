import dspy 
from pydantic import BaseModel, Field
from chroma import qdrant
import os

llm = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

dspy.settings.configure(lm = llm)

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

class QuizRAG(dspy.Module):
    def __init__(self):
        super().__init__() 
        self.generate_quiz = dspy.ChainOfThought(QuizSignature)
    def forward(self, quiz_text):
        context = qdrant.search(
            query=quiz_text,
            search_type="similarity_score_threshold"
        )
        context_text = []
        for doc in context:
            context_text.append(str(doc.page_content))
        quiz_input = QuizInput(topic=str(quiz_text), context=context_text)
        prediction = self.generate_quiz(input=quiz_input)
        return prediction
