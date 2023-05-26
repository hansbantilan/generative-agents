import os
import openai
import streamlit as st
from streamlit_chat import message

from generative_agents.modeling.agent_helpers import load_llm, load_pinecone, load_agent
from generative_agents.utility import logger, well_known_paths
from generative_agents.utility.utility import load_params
from generative_agents.modeling.interactions import talk_to_penpal, start_conversation_w_penpal

log = logger.init("Streamlit app")


def generate_response(user_input):
    res = talk_to_penpal(agent, user_input, params)
    return res



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

if 'generated' not in st.session_state:
    st.session_state["generated"] = [start_conversation_w_penpal(agent, params, "Coffee Shop")]

if "history" not in st.session_state:
    st.session_state["history"] = []


def get_text():
    input_text = st.text_input("Student [enter your message here]:", "")
    return input_text



### Chat interface on streamlit app ###
initial_question = params['user_input']
log.info(f"Asking {params['name']} the following question...\n{params['user_input']}")
message(st.session_state["generated"][0], key=str(-1))

user_input = get_text()

if user_input:
    output = generate_response(user_input)
    st.session_state.history.append(user_input)
    st.session_state.generated.append(output)


# print(st.session_state["generated"])
# print(st.session_state["history"])

if st.session_state["history"]:
    for i in range(len(st.session_state["history"]), 0, -1):
        message(st.session_state["history"][i-1], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))


