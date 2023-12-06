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
    instructions="You help with tour selection and tour consultation at a company that sells educational tours. gl_profiles.json gives information about the customers who are teachers, and subject name is what the teacher teaches. Some of the teachers are repeating customers and repeat_gls_hist.json shows their past trips. tours_data.json shows the different tourcodes available and their details.",
    tools=[{"type": "retrieval"}, {"type": "code_interpreter"}],
    file_ids=[tours_data.id, gl_profiles_data.id, repeat_gls_hist_data.id],
)
show_json(assistant)

# prompt for top-10 recommendations for each GL
thread, run = create_thread_and_run(
    "Based on the tours that customers with id_0 to id_100 has participated in in the past (from the file repeat_gls_hist.json) and their subject name, for each customer recommend 10 tours with tourcode and tournames . Output the list with individual id and tourcodes in json format.",
    assistant,
)
run = wait_on_run(run, thread)
show_response(thread)
show_messages(thread)
