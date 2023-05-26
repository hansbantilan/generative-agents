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

st.sidebar.write("*Customize your chatbot*")
customization_options = load_params(os.path.join(
            well_known_paths["DEMO_DIR"],
            "customization.yaml",
        ))

customization_options_changed = False

cefr_level = st.sidebar.selectbox("CEFR Level", customization_options["cefr_levels"])
meet_location = st.sidebar.selectbox("meet_location", customization_options["meet_locations"])

if "meet_location" not in st.session_state:
    st.session_state["meet_location"] = meet_location

if meet_location != st.session_state["meet_location"]:
    customization_options_changed = True
    st.session_state["meet_location"] = meet_location

params = load_params(
        os.path.join(
            well_known_paths["PARAMS_DIR"],
            "penpal_agent",
            f"default.yaml",
        )
    )

params['level'] = cefr_level # override default level with user customization


### Setup agent ###
@st.cache_resource
def setup_agent(cefr_level):
    llm = load_llm()

    if params["is_pinecone"]:
        load_pinecone()

    agent = load_agent(params, llm)
    return agent

agent = setup_agent(cefr_level)

if 'generated' not in st.session_state:
    st.session_state["generated"] = []

if len(st.session_state["generated"]) == 0 or customization_options_changed:
    initial_agent_response = start_conversation_w_penpal(agent, params, meet_location)
    st.session_state["generated"].append(initial_agent_response)
    customization_options_changed = False

if "history" not in st.session_state:
    st.session_state["history"] = []


styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

def get_text():
    input_text = st.text_input("Student [enter your message here]:", "")
    return input_text


### Chat interface on streamlit app ###
message(st.session_state["generated"][0], key=str(-1))

user_input = get_text()

if user_input:
    output = generate_response(user_input)
    st.session_state.history.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["history"]:
    for i in range(0, len(st.session_state["history"])):
        message(st.session_state["history"][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i+1], key=str(i))
