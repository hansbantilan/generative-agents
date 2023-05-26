import os
import openai
import streamlit as st
from streamlit_chat import message

from generative_agents.modeling.agent_helpers import load_llm, load_pinecone, load_agent
from generative_agents.utility import logger, well_known_paths
from generative_agents.utility.utility import load_params
from generative_agents.modeling.interactions import ask_penpal

log = logger.init("Streamlit app")

def main():
    log.info(f"Asking {params['name']} the following question...\n{params['question']}")
    print(ask_penpal(agent, params))

def generate_response(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )
    message = res.choices[0].message.content
    return message


st.title("EF PenPal 🤖 ")
st.sidebar.title("EF PenPal")
st.sidebar.write("This is a demo of the EF Penpal Chatbot built with Langchain + Generative Agents.")

params = load_params(
        os.path.join(
            well_known_paths["PARAMS_DIR"],
            "penpal_agent",
            f"default.yaml",
        )
    )

if 'generated' not in st.session_state:
    st.session_state["generated"] = [params['question']]

if "history" not in st.session_state:
    st.session_state["history"] = []


def get_text():
    input_text = st.text_input("Student [enter your message here]:", "")
    return input_text



### Setup agent ###
@st.cache_resource
def setup_agent():
    llm = load_llm()
    log.info(f"Loading Parameters...")

    if params["is_pinecone"]:
        load_pinecone()

    agent = load_agent(params, llm)
    return agent

agent = setup_agent()

### Chat interface on streamlit app ###
initial_question = params['question']
log.info(f"Asking {params['name']} the following question...\n{params['question']}")
message(st.session_state["generated"][0], key=str(-1))

user_input = get_text()

if user_input:
    # output = generate_response(user_input)
    output = "test"
    st.session_state.history.append(user_input)
    st.session_state.generated.append(output)


# print(st.session_state["generated"])
# print(st.session_state["history"])

if st.session_state["history"]:
    for i in range(len(st.session_state["history"]), 0, -1):
        message(st.session_state["history"][i-1], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))


