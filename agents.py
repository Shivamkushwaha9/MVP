import requests, os
import streamlit as st
from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv('google_api_key')
# google_api_key = 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", verbose=True, temperature=0.2, google_api_key=google_api_key
)


def fetch_raw_content(repo_url, file_path):
    url_parts = repo_url.replace("https://github.com/", "").split("/")
    owner = url_parts[0]
    repo = url_parts[1]
    branch = "main"  # Assuming the default branch is 'main'
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    
    response = requests.get(raw_url)
    if response.status_code == 200:
        return response.text
    else:
        st.error(f"Failed to retrieve raw content. Status code: {response.status_code}")
        return None
    
    
def generate_questions(raw_content):
    interviewer = Agent(
        role="Interviewer",
        goal=f"You are interviewer, Ask 5 coding questions about {raw_content} and then ask some theory questions. You don't have to give the answers.",
        backstory="""You're an expert senior interviewer for machine learning jobs at a large company with years of experience in the domain and a deep knowledge of python and machine learning concepts such as retrieval augmented generation.
        You're responsible for asking relevant questions of great value in order to hire the best candidate based on the given code.""",
        allow_delegation=False,
        llm=llm,
        cache=True,
    )

    interview_task = Task(
        description=f"Create mix of coding and theory questions based on the codebase {raw_content} Also, The coding questions must be the questions which user have to answer by writing the code. Just ask the questions, no need to mention the answers or topics from which they are asked based on the code provided.",
        agent=interviewer,
        expected_output=f"Interview questions based on the following codebase: {raw_content} which must consist of 5-10 coding or theory questions based on the codebase provided"
    )

    recruiting_crew = Crew(
        agents=[interviewer],
        tasks=[interview_task],
        verbose=False,
        process=Process.sequential
    )

    return recruiting_crew.kickoff()


def evaluate_answers(questions, answers):
    evaluator_agent = Agent(
        role="Evaluator",
        goal=f"You will have to evaluate the responses/answers given to you which will be based on the 5 coding questions and 5 theoretical questions. The questions will be asked based on the projects which the interviewee has worked on",
        backstory="Seasoned programmer with 15 years of programming and writing efficient Python and you are specialized in machine learning.",
        allow_delegations=False,
        llm=llm,
        cache=True
    )

    evaluation_task = Task(
        description=f"Evaluate whether the following code/theoretical answers: {answers} provided by the user are right or wrong based on the predefined questions: {questions}. Consider the relevance of questions and answers, and rate how detailed and well-answered they are.You don't have to generate answers, you have to just evaluate them",
        agent=evaluator_agent,
        expected_output="Detailed evaluation with a rough estimate based on the relevance of the questions and the answers given."
    )

    evaluation_crew = Crew(
        agents=[evaluator_agent],
        tasks=[evaluation_task],
        verbose=False,
        process=Process.sequential
    )

    return evaluation_crew.kickoff()