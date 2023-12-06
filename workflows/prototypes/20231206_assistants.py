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

# Create Assistant
assistant = client.beta.assistants.create(
    name="ToursGPT",
    model="gpt-4-1106-preview",
    instructions="You help with tour selection and tour consultation at a company that sells educational tours.",
    tools=[{"type": "retrieval"}],
    file_ids=[tours_data.id, gl_profiles_data.id, repeat_gls_hist_data.id],
)
show_json(assistant)

thread, run = create_thread_and_run(
    "What data does repeat_gls_hist.json have?", assistant
)
run = wait_on_run(run, thread)
show_response(thread)
show_messages(thread)
