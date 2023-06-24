import os

import streamlit as st
from streamlit_chat import message

from generative_agents.modeling.agent_helpers import load_agent, load_llm, load_pinecone
from generative_agents.modeling.interactions import (
    start_conversation_w_penpal,
    talk_to_penpal,
)
from generative_agents.utility import logger, well_known_paths
from generative_agents.utility.utility import get_completion, load_params

log = logger.init("Streamlit app")


def generate_response(user_input):
    res = talk_to_penpal(agent, user_input, params)
    return res


st.title("EF PenPal 🤖 ")
st.sidebar.title("EF PenPal")
st.sidebar.write(
    "This is a demo of the EF Penpal Chatbot built with ChatGPT, Langchain + Generative Agents."
)
st.sidebar.write("Authors:\n Hans Bantilan, Amrita Panesar")
st.sidebar.write("*Customize your penpal*")
customization_options = load_params(
    os.path.join(
        well_known_paths["PENPAL_APP_DIR"],
        "customization.yaml",
    )
)

if "customization_options_changed" not in st.session_state:
    st.session_state.customization_options_changed = False

params = load_params(
    os.path.join(
        well_known_paths["PARAMS_DIR"],
        "penpal_agent",
        f"default.yaml",
    )
)

llm_type = st.sidebar.selectbox(
    label="LLM Type",
    options=customization_options["llm_types"],
    index=customization_options["llm_types"].index(
        customization_options["default_llm_type"]
    ),
)
name = st.sidebar.text_input(
    label="Name",
    value=customization_options["default_name"],
)
cefr_level = st.sidebar.selectbox(
    label="CEFR Level",
    options=customization_options["cefr_levels"],
    index=customization_options["cefr_levels"].index(
        customization_options["default_cefr_level"]
    ),
)
meet_location = st.sidebar.selectbox(
    label="Location of meetup",
    options=customization_options["meet_locations"],
    index=customization_options["meet_locations"].index(
        customization_options["default_meet_location"]
    ),
)


def update_customization_options():
    st.session_state["llm_type"] = llm_type
    st.session_state["name"] = name
    st.session_state["cefr_level"] = cefr_level
    st.session_state["meet_location"] = meet_location
    st.session_state["customization_options_changed"] = True
    st.session_state["input_text"] = ""


update_penpal_btn = st.sidebar.button(
    "Create Penpal",
    help="Click to customize your penpal.",
    on_click=update_customization_options,
)

if "llm_type" not in st.session_state:
    st.session_state["llm_type"] = customization_options["default_llm_type"]

if "name" not in st.session_state:
    st.session_state["name"] = customization_options["default_name"]

if "cefr_level" not in st.session_state:
    st.session_state["cefr_level"] = customization_options["default_cefr_level"]

if "meet_location" not in st.session_state:
    st.session_state["meet_location"] = customization_options["default_meet_location"]

params["name"] = st.session_state["name"]
params["level"] = st.session_state[
    "cefr_level"
]  # override default level with user customization


### Setup agent ###
# @st.cache_resource #HB (don't cache this for now, as we test llm_type)
def setup_agent(params):
    log.info(f"params: {params}")
    llm = load_llm(llm_type)
    if params["is_pinecone"]:
        load_pinecone()
    agent = load_agent(params, llm)
    return agent


agent = setup_agent(params)

if "generated" not in st.session_state:
    st.session_state["generated"] = []

# start conversation if app just loaded or customization options for agent have been changed
if (
    len(st.session_state["generated"]) == 0
    or st.session_state.customization_options_changed
):
    st.session_state["language_level_history"] = []
    st.session_state["history"] = []
    st.session_state["generated"] = []
    log.info("Starting conversation...")
    initial_agent_response = start_conversation_w_penpal(agent, params, meet_location)
    st.session_state.generated.append(initial_agent_response)
    st.session_state.customization_options_changed = False


if "history" not in st.session_state:
    st.session_state["history"] = []

if "language_level_history" not in st.session_state:
    st.session_state["language_level_history"] = []

styl = f"""
<style>
    .main .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
    .main .stButton {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

st.write(f"You are talking to your penpal {st.session_state['name']}.")

### Chat interface on streamlit app ###
message(st.session_state["generated"][0], key=str(-1))

col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Student [enter your message here]:", "", key="input_text"
    )

with col2:
    send_button = st.button("Send", key="send_button")

if send_button:
    output = generate_response(user_input)
    # get language level of student thought by ChatGPT
    language_level_prompt = f"""
        You are a language teacher grading a student's English level from a written response delimited by triple backticks. You should rate the level of the response from 1 to 16, with 1 being very basic and 16 being advanced.
        Student's written response:
        {user_input}

        Output a single number from 1 to 16 as the answer:
    """
    log.info("Language level prompt: " + language_level_prompt)
    language_level = get_completion(language_level_prompt)
    language_level = language_level.split(" ")[-1]
    if language_level.isdigit():
        st.session_state.language_level_history.append(language_level)
    else:
        st.session_state.language_level_history.append("Cannot compute language level")

    st.session_state.history.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["history"]:
    for i in range(0, len(st.session_state["history"])):
        message(st.session_state["history"][i], is_user=True, key=str(i) + "_user")
        curr_language_level = st.session_state.language_level_history[i]
        # output if language level has significantly changed
        if i == 0:
            st.write(f"Current language level: {curr_language_level}")
        else:
            prev_language_level = st.session_state.language_level_history[i - 1]
            if prev_language_level < curr_language_level:
                st.write(
                    f"Your language level has increased from {prev_language_level} to {curr_language_level}"
                )
            elif prev_language_level > curr_language_level:
                st.write(
                    f"Your language level has decreased from {prev_language_level} to {curr_language_level}"
                )
            else:
                st.write(
                    f"Your language level has stayed the same at {curr_language_level}"
                )

        message(st.session_state["generated"][i + 1], key=str(i))
