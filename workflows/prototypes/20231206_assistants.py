import ast
import json
import os
import time

import openai


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )


def create_thread_and_run(user_message, assistant):
    thread = client.beta.threads.create()
    run = submit_message(assistant.id, thread, user_message)
    return thread, run


def continue_thread_and_run(thread, user_message, assistant):
    run = submit_message(assistant.id, thread, user_message)
    return thread, run


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def show_json(obj):
    display(json.loads(obj.model_dump_json()))


def show_messages(thread):
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    show_json(messages)


def show_response(thread):
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


# Instantiate OpenAI client
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

# Upload the file
tours_data = client.files.create(
    file=open(
        "tours_data.json",
        "rb",
    ),
    purpose="assistants",
)
gl_profiles_data = client.files.create(
    file=open(
        "gl_profiles.json",
        "rb",
    ),
    purpose="assistants",
)
repeat_gls_hist_data = client.files.create(
    file=open(
        "repeat_gls_hist.json",
        "rb",
    ),
    purpose="assistants",
)
repeat_gls_hist_data = client.files.create(
    file=open(
        "repeat_gls_hist.json",
        "rb",
    ),
    purpose="assistants",
)


# Create Assistant
assistant = client.beta.assistants.create(
    name="TC-Assistant",
    model="gpt-4-1106-preview",
    instructions="Based on the tours that customers with id_0 to id_99 has participated in in the past (from the file repeat_gls_hist.json) and their subject name, for each customer recommend 10 tours with tourcode and tournames. Don't produce same tours for everybody, you can be creative. You should produce recommendations for all of 100 customers. Output the list with individual id and tourcodes in json format.",
    tools=[{"type": "retrieval"}, {"type": "code_interpreter"}],
    file_ids=[tours_data.id, gl_profiles_data.id, repeat_gls_hist_data.id],
)
show_json(assistant)

# prompt for top-10 recommendations for each GL
thread, run1 = create_thread_and_run(
    "Based on the tours that customers with id_0 to id_99 has participated in in the past (from the file repeat_gls_hist.json) and their subject name, for each customer recommend 10 tours with tourcode and tournames. Don't produce same tours for everybody, you can be creative. You should produce recommendations for all of 100 customers. Output the list with individual id and tourcodes in csv format, where every row should be id, and every column should be tourcode. If you recommend a tour for a customer that cell should be 1, otherwise 0.",
    assistant,
)
run1 = wait_on_run(run1, thread)
show_response(thread)

# prompt for a downloadable file
thread, run2 = continue_thread_and_run(
    thread,
    "Please provide the full json recommendation data for all customers with IDs ranging from `id_0` to `id_99`. Output in the file_id of the resulting file.",
    assistant,
)
run2 = wait_on_run(run2, thread)
show_response(thread)

# retrieve file contents
messages = client.beta.threads.messages.list(thread_id=thread.id)
for message in messages:
    file_id = message.file_ids[0]
    break
results_str = client.files.retrieve_content(file_id)
results_dict = ast.literal_eval(results_str)
