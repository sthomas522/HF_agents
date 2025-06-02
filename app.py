import os
import gradio as gr
import requests
import inspect
import pandas as pd
import traceback
import re
import pandas as pd
from huggingface_hub import login
from tools import create_memory_safe_workflow, get_file_type, write_bytes_to_temp_dir, AgentState, extract_final_answer, run_agent

import re

def strip_final_answer(text):
    """
    Removes 'FINAL ANSWER:' (case-insensitive) and all following whitespace from the start of the string.
    Returns the remainder of the string.
    """
    # The regex matches 'FINAL ANSWER:', optional colon, and all whitespace after it
    return re.sub(r'^\s*FINAL ANSWER:\s*', '', text, flags=re.IGNORECASE)

# Example usage:
s = "FINAL ANSWER: Joe Torre"
print(strip_final_answer(s))  # Output: Joe Torre

s2 = "  FINAL ANSWER:   Jane Doe"
print(strip_final_answer(s2))  # Output: Jane Doe


def print_answers_dataframe(answers_payload):
    # Create a list of question numbers from 1 to length of answers_payload
    question_numbers = list(range(1, len(answers_payload) + 1))
    
    # Extract task_id and submitted_answer from the list of dictionaries
    task_ids = [item["task_id"] for item in answers_payload]
    submitted_answers = [item["submitted_answer"] for item in answers_payload]
    
    # Create the DataFrame
    df = pd.DataFrame({
        "question_number": question_numbers,
        "task_id": task_ids,
        "submitted_answer": submitted_answers
    })
    
    # Print the DataFrame
    print(df)

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# login(token=os.environ["HF_TOKEN"])

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
#class BasicAgent:
#    def __init__(self):
#        print("BasicAgent initialized.")
#    def __call__(self, question: str) -> str:
#        print(f"Agent received question (first 50 chars): {question[:50]}...")
#        fixed_answer = "This is a default answer."
#        print(f"Agent returning fixed answer: {fixed_answer}")
#        return fixed_answer

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    files_url = f"{api_url}/files"

    # 1. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        hf_questions = response.json()
        if not hf_questions:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(hf_questions)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None
    
    # 2. Create states
    try:
        for item in hf_questions:
            file_name = item.get('file_name', '')
            if file_name == '':
                item['input_file'] = None
                item['file_type'] = None
                item['file_path'] = None
            else:
                # Call the API to retrieve the file; adjust params as needed
                task_id = item['task_id']
                api_response = requests.get(f"{files_url}/{task_id}")
                print(f"api_response = {api_response.status_code}")
                if api_response.status_code == 200:
                    item['input_file'] = api_response.content  # Store file as bytes
                    item['file_type'] = get_file_type(file_name)
                    item['file_path'] = write_bytes_to_temp_dir(item['input_file'], file_name)
                else:
                    item['input_file'] = None  # Or handle error as needed
                    item['file_type'] = None
                    item['file_path'] = None
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error creating new states: {tb_str}")
        return f"Error creating new states: {tb_str}", None
    
    agent = create_memory_safe_workflow()    

    # Setup states for questions and run agent
    answers_payload = []
    results_log = []
    for r in range(len(hf_questions)):
        s = AgentState(question = hf_questions[r]['question'],
                    input_file = hf_questions[r]['input_file'],
                    file_type = hf_questions[r]['file_type'],
                    file_path = hf_questions[r]['file_path'])
        try:
            task_id = hf_questions[r]['task_id']
            question_text = hf_questions[r]['question']
            full_answer = run_agent(agent, s)
            submitted_answer = extract_final_answer(full_answer[-1].content)
            print(f"\n\nQuestion {r+1} Answer: {submitted_answer}\n\n")
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)
            
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.\n\n")
        print("Full answer list\n")
        print_answers_dataframe(answers_payload=answers_payload)
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    

# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# HF Course Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for HF Intelligent Agent Evaluation...")
    demo.launch(debug=True, share=False)