import streamlit as st
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process


#For Sub-Process
import os
import subprocess
import sys
import re
import cv2
from threading import Thread


#Function to runa
# from gaze_tracking import cv_script



#Env path for CV
cv_env_path = r"C:\Users\SHIVAM\PycharmProjects\Face_rec\cv_env"

#Script path of Executable
cv_script_path = r"C:\Users\SHIVAM\Desktop\GazeTracking\gaze_tracking\gaze_tracking.py"

#Yaha env activate kr raha
def activate_venv(env_path):
    activate_script = os.path.join(env_path, 'Scripts', 'activate.bat')
    return f'call "{activate_script}" && '

#here for runnning the scripts
def run_script(script_path, env_path):
    python_path = os.path.join(env_path, 'Scripts', 'python.exe')
    command = f'{activate_venv(env_path)} "{python_path}" "{script_path}"'
    subprocess.run(command, shell=True)
    
def display_cv_window(image_placeholder):
    from gaze_tracking import cv_script

    while True:
        frame = cv_script.get_frame()

        if frame is None:
            st.error("Failed to capture image")
            break

        # Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame
        image_placeholder.image(frame, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






# Google API key
google_api_key = "AIzaSyCY5RigB8HjxAsCmyagpfY-GfNOM5u790g"
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.9, google_api_key=google_api_key
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

#Creating the image placeholder
# image_placeholder = st.empty()
# comp_v = run_script(cv_script_path,cv_env_path)
def main():
    st.title("AI Agent MVP")
    
    cv_column, _ = st.columns([1, 3])

    with cv_column:
        st.write("CV Window:")
        image_placeholder = st.empty()
        
        cv_thread = Thread(target=run_script, args=(cv_script_path, cv_env_path))
        cv_thread.start()
        display_thread = Thread(target=display_cv_window, args=(image_placeholder,))
        display_thread.start()
    
    # image_placeholder = st.empty()
    # comp_v = run_script(cv_script_path,cv_env_path)
    # image_placeholder.image(comp_v,channels="RGB")
    
    

    if 'stage' not in st.session_state:
        st.session_state.stage = 'input'

    if st.session_state.stage == 'input':
        repo_url = st.text_input("Enter the GitHub repository URL:")
        file_path = st.text_input("Enter the main file path:")

        if st.button("Fetch Raw Content and Generate Questions"):
            raw_content = fetch_raw_content(repo_url, file_path)
            if raw_content:
                with st.spinner("Generating questions..."):
                    st.session_state.questions = generate_questions(raw_content)
                st.session_state.stage = 'answer'
                st.experimental_rerun()

    elif st.session_state.stage == 'answer':
        st.subheader("Interview Questions and Answers")
        
        questions = st.session_state.questions.split('\n')
        st.session_state.answers = {}

        for i, question in enumerate(questions, 1):
            if question.strip():
                st.write(f"Q{i}: {question}")
                answer = st.text_area(f"Your Answer for Q{i}", key=f"answer_{i}", height=100)
                st.session_state.answers[f'Answer_{i}'] = answer

        if st.button("Submit All Answers"):
            if all(st.session_state.answers.values()):
                st.session_state.stage = 'evaluate'
                st.experimental_rerun()
            else:
                st.warning("Please answer all questions before submitting.")

    elif st.session_state.stage == 'evaluate':
        st.subheader("Evaluation")
        with st.spinner("Evaluating answers..."):
            evaluation = evaluate_answers(st.session_state.questions, st.session_state.answers)
        st.write(evaluation)

        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

if __name__ == "__main__":
    main()