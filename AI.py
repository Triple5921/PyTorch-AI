import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json
import os
import re
from datetime import datetime
from PIL import Image, ImageDraw
from huggingface_hub import InferenceClient

# Optional Imports (Handle missing libraries gracefully)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# --- 1. CONFIGURATION ---
MEMORY_FILE = "marin_organic_mem.json"
BRAIN_FILE = "marin_organic_brain.pth"
PERSONALITY_FILE = "Personality.txt"
DIALOG_FILE = "Dialog Example.txt"
IMG_FOLDER = "Ai_Pictures"

if not os.path.exists(IMG_FOLDER): os.makedirs(IMG_FOLDER)

# --- 2. EVOLVING NEURAL ARCHITECTURE ---
class OrganicBrain(nn.Module):
    def __init__(self, input_size=12, hidden_size=16, output_size=6):
        super(OrganicBrain, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.activation(self.layer3(x))
        return x

    def grow_brain(self):
        old_hidden = self.hidden_size
        new_hidden = old_hidden + 4 
        new_l1 = nn.Linear(self.input_size, new_hidden)
        new_l2 = nn.Linear(new_hidden, new_hidden)
        new_l3 = nn.Linear(new_hidden, self.output_size)
        with torch.no_grad():
            new_l1.weight[:old_hidden, :] = self.layer1.weight
            new_l1.bias[:old_hidden] = self.layer1.bias
            new_l2.weight[:old_hidden, :old_hidden] = self.layer2.weight
            new_l2.bias[:old_hidden] = self.layer2.bias
            new_l3.weight[:, :old_hidden] = self.layer3.weight
            new_l3.bias[:] = self.layer3.bias
        self.layer1 = new_l1
        self.layer2 = new_l2
        self.layer3 = new_l3
        self.hidden_size = new_hidden
        return f"Neural Architecture Expanded: {old_hidden} -> {new_hidden} Hidden Nodes"

# --- 3. SENSORY INPUT ---
def extract_sensory_data(text):
    text = text.lower()
    joy_set = {"happy", "good", "great", "calm", "centered", "content", "peaceful", "serene", "bliss", "ecstatic", "thrilled", "radiant", "playful", "passionate", "alive", "grateful", "excited", "inspired", "confident", "safe", "optimistic"}
    sad_set = {"sad", "bad", "depressed", "grief", "despair", "hopeless", "lonely", "isolated", "empty", "exhausted", "weary", "broken", "heartbroken", "gloomy", "miserable", "hurt", "disappointed", "sorrow", "blue"}
    anger_set = {"angry", "mad", "hate", "furious", "irate", "annoyed", "irritated", "agitated", "bitter", "resentful", "hostile", "pissed", "rage", "frustrated", "disgusted", "contempt", "cranky", "upset", "vindictive"}
    fear_set = {"scared", "afraid", "fear", "terrified", "panic", "anxious", "nervous", "worried", "stressed", "tense", "uneasy", "frightened", "paralyzed", "shaken", "threatened", "trapped", "vulnerable", "overwhelmed"}
    shame_set = {"ashamed", "shame", "guilty", "guilt", "embarrassed", "humiliated", "mortified", "regret", "remorse", "sorry", "worthless", "stupid", "foolish", "inadequate", "shy"}
    love_set = {"love", "loving", "loved", "connected", "affectionate", "caring", "tender", "warm", "touched", "close", "intimate", "cherished", "adored", "supported", "appreciated"}
    phys_set = {"achy", "numb", "dizzy", "pain", "shaky", "sweaty", "cold", "hot", "burning", "tingling", "tense", "tight", "breathless", "suffocated", "sick", "tired", "sleepy", "sore", "heavy", "drained"}
    conf_set = {"confused", "lost", "uncertain", "unsure", "perplexed", "puzzled", "doubt", "weird", "strange", "what", "huh"}
    shock_set = {"shocked", "amazed", "surprised", "stunned", "wow", "omg", "whoa", "unbelievable", "incredible"}

    def score(word_set): return 1.0 if any(w in text for w in word_set) else 0.0
    vec = [score(joy_set), score(sad_set), score(anger_set), score(fear_set), score(shame_set), score(love_set), score(phys_set), score(conf_set), score(shock_set), 1.0 if text.isupper() or text.count("!") > 1 else 0.2, min(len(text.split()) / 25.0, 1.0), 1.0 if "?" in text else 0.0]
    return torch.tensor(vec)

# --- 4. IMAGE GENERATOR ---
def generate_image(prompt):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{IMG_FOLDER}/marin_gen_{timestamp}.png"
    img = Image.new('RGB', (512, 512), color = (25, 20, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([(10,10), (502,502)], outline="#00D1FF", width=4)
    d.text((50, 200), f"VISUALIZATION:\n{prompt[:150]}...", fill=(200, 255, 255))
    d.text((50, 450), "Rendering engine: Placeholder", fill="#00D1FF")
    img.save(filename)
    return filename

# --- 5. UI COMPONENTS ---
def draw_hud(inputs, traits, brain):
    node_color = st.session_state.settings.get("node_color", "#E91E63")
    node_size = st.session_state.settings.get("node_size", 8)
    st.markdown(f"<style>div[data-testid='stVerticalBlock'] > div:has(div.sticky-container) {{ position: sticky; top: 0; background-color: rgba(10, 10, 10, 0.95); z-index: 999; padding: 5px; border-bottom: 1px solid {node_color}; backdrop-filter: blur(10px); }}</style><div class='sticky-container'></div>", unsafe_allow_html=True)
    with st.container():
        st.caption("🧠 NEURAL DIAGNOSTICS")
        c1, c2, c3 = st.columns([1.5, 1, 1])
        with c1:
            hidden_count = brain.hidden_size
            layers = [12, hidden_count, hidden_count, 6]
            edge_x, edge_y, node_x, node_y = [], [], [], []
            for i, size in enumerate(layers):
                for j in range(size):
                    node_x.append(i * 3); node_y.append(j - (size - 1) / 2)
            for i in range(len(layers) - 1):
                for j in range(layers[i]):
                    for k in range(layers[i+1]):
                        edge_x += [i*3, (i+1)*3, None]; edge_y += [j-(layers[i]-1)/2, k-(layers[i+1]-1)/2, None]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#444'), hoverinfo='none'))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=node_size, color=node_color)))
            fig.update_layout(title=f"Network ({hidden_count} Nodes)", height=150, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        with c2:
            fig = go.Figure(go.Scatterpolar(r=inputs[:6], theta=['Joy', 'Sadness', 'Anger', 'Fear', 'Shame', 'Love'], fill='toself', line=dict(color='#00FF99'), fillcolor='rgba(0,255,153,0.3)'))
            fig.update_layout(title="Sensory Input", height=150, margin=dict(l=25,r=25,t=30,b=20), polar=dict(radialaxis=dict(visible=False, range=[0,1]), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        with c3:
            fig = go.Figure(go.Scatterpolar(r=traits, theta=['Energy', 'Openness', 'Empathy', 'Attraction', 'Anxiety', 'Mood'], fill='toself', line=dict(color='#FF69B4'), fillcolor='rgba(255,105,180,0.3)'))
            fig.update_layout(title="Brain State", height=150, margin=dict(l=25,r=25,t=30,b=20), polar=dict(radialaxis=dict(visible=False, range=[0,1]), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        st.divider()

# --- 6. SYSTEM OPS ---
def save_system():
    if isinstance(st.session_state.last_inputs, (torch.Tensor, np.ndarray)): input_list = st.session_state.last_inputs.tolist()
    else: input_list = st.session_state.last_inputs
    if isinstance(st.session_state.traits, (torch.Tensor, np.ndarray)): traits_list = st.session_state.traits.tolist()
    else: traits_list = st.session_state.traits
    data = {"messages": st.session_state.messages, "traits": traits_list, "last_inputs": input_list, "density": st.session_state.density, "hidden_size": st.session_state.brain.hidden_size, "settings": st.session_state.settings}
    with open(MEMORY_FILE, "w") as f: json.dump(data, f)
    torch.save(st.session_state.brain.state_dict(), BRAIN_FILE)

def load_system():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f: data = json.load(f)
            st.session_state.messages = data.get("messages", [])
            st.session_state.traits = data.get("traits", [0.5]*6)
            st.session_state.last_inputs = data.get("last_inputs", [0.0]*12)
            st.session_state.density = data.get("density", 0.0)
            st.session_state.settings.update(data.get("settings", {}))
            saved_hidden = data.get("hidden_size", 16)
            if saved_hidden != st.session_state.brain.hidden_size:
                st.session_state.brain = OrganicBrain(12, saved_hidden, 6)
            if os.path.exists(BRAIN_FILE):
                try: st.session_state.brain.load_state_dict(torch.load(BRAIN_FILE))
                except: pass
        except: pass

def load_text_files():
    p, d = "You are Marin.", ""
    if os.path.exists(PERSONALITY_FILE): 
        with open(PERSONALITY_FILE, "r", encoding="utf-8") as f: p = f.read()
    if os.path.exists(DIALOG_FILE):
        with open(DIALOG_FILE, "r", encoding="utf-8") as f: d = f.read()
    return p, d

# --- 7. INITIALIZATION ---
if 'initialized' not in st.session_state:
    st.session_state.brain = OrganicBrain()
    st.session_state.traits = [0.5]*6
    st.session_state.last_inputs = [0.0]*12
    st.session_state.messages = []
    st.session_state.density = 0.0
    st.session_state.notifications = []
    st.session_state.show_diagnostics = True
    st.session_state.show_thoughts = False 
    st.session_state.settings = {"learning_rate": 0.05, "img_gen": True, "response_len": "Short & Punchy", "status_text": "Marin is thinking...", "node_color": "#E91E63", "node_size": 8, "backend": "Cloud API (HuggingFace)"}
    load_system()
    st.session_state.optimizer = optim.Adam(st.session_state.brain.parameters(), lr=st.session_state.settings["learning_rate"])
    st.session_state.initialized = True
    st.session_state.processing = False

# --- 8. UI RENDER ---
st.set_page_config(page_title="Marin OS 9.0", layout="wide")
st.markdown("<style>.stApp { background-color: #0E0E0E; color: #E0E0E0; } .stButton>button { border: 1px solid #FF69B4; color: #FF69B4; width: 100%; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings")
    
    # --- UNIVERSAL BACKEND SELECTOR ---
    backend_options = ["Cloud API (HuggingFace)"]
    if OPENAI_AVAILABLE: backend_options.insert(0, "OpenAI (GPT-4o)")
    if OLLAMA_AVAILABLE: backend_options.append("Local Brain (Ollama)")
    
    backend_choice = st.radio("AI Backend:", backend_options, index=0)
    st.session_state.settings["backend"] = backend_choice
    
    # --- DYNAMIC TOKEN INPUT ---
    if "OpenAI" in backend_choice:
        openai_key = st.text_input("Enter OpenAI Key (sk-...):", type="password", key="openai_key_input")
        if openai_key:
            st.session_state["openai_key"] = openai_key
            st.success("✅ OpenAI Key Active")
    
    elif "HuggingFace" in backend_choice:
        # Check secrets first, then manual input
        try:
            hf_secret = st.secrets["HF_TOKEN"]
            st.success("✅ HF Key found in Secrets")
        except:
            hf_key = st.text_input("Enter Hugging Face Token (hf_...):", type="password", key="hf_key_input")
            if hf_key:
                st.session_state["hf_key"] = hf_key
                st.success("✅ HF Key Active")
    
    if st.button("Toggle Diagnostics HUD"): st.session_state.show_diagnostics = not st.session_state.show_diagnostics
    if st.button("Toggle Inner Monologue"): st.session_state.show_thoughts = not st.session_state.show_thoughts
    
    st.divider()
    if st.button("⚠️ Factory Reset"):
        if os.path.exists(MEMORY_FILE): os.remove(MEMORY_FILE)
        if os.path.exists(BRAIN_FILE): os.remove(BRAIN_FILE)
        st.session_state.messages = []
        st.rerun()

tab1, tab2 = st.tabs(["💬 Terminal Link", "⚙️ System Settings"])

with tab2:
    st.header("Neural Configuration")
    col_c, col_s = st.columns(2)
    with col_c:
        new_color = st.color_picker("Neural Node Color", st.session_state.settings["node_color"])
        if new_color != st.session_state.settings["node_color"]: st.session_state.settings["node_color"] = new_color; save_system(); st.rerun()
    with col_s:
        new_size = st.slider("Neural Node Size", 4, 20, st.session_state.settings["node_size"])
        if new_size != st.session_state.settings["node_size"]: st.session_state.settings["node_size"] = new_size; save_system(); st.rerun()
    st.divider()
    st.subheader("Logic Core")
    new_lr = st.slider("Neuroplasticity Rate", 0.01, 0.50, st.session_state.settings["learning_rate"])
    if new_lr != st.session_state.settings["learning_rate"]: st.session_state.settings["learning_rate"] = new_lr; st.session_state.optimizer = optim.Adam(st.session_state.brain.parameters(), lr=new_lr); save_system()
    new_len = st.selectbox("Response Verbosity", ["Short & Punchy", "Detailed & Expressive"], index=0 if st.session_state.settings["response_len"] == "Short & Punchy" else 1)
    if new_len != st.session_state.settings["response_len"]: st.session_state.settings["response_len"] = new_len; save_system()
    new_img = st.checkbox("Enable Image Generation Module", value=st.session_state.settings["img_gen"])
    if new_img != st.session_state.settings["img_gen"]: st.session_state.settings["img_gen"] = new_img; save_system()
    new_status = st.text_input("Custom Loading Status", value=st.session_state.settings["status_text"])
    if new_status != st.session_state.settings["status_text"]: st.session_state.settings["status_text"] = new_status; save_system()

with tab1:
    if st.session_state.show_diagnostics: draw_hud(st.session_state.last_inputs, st.session_state.traits, st.session_state.brain)
    if st.session_state.notifications:
        for note in st.session_state.notifications: st.success(note)
        st.session_state.notifications = []
    
    if st.session_state.show_thoughts:
        chat_col, thought_col = st.columns([1.5, 1])
        with chat_col: st.caption("💬 Conversation")
        with thought_col: st.caption("🧠 Inner Monologue (Read Only)")
    else:
        chat_col = st.container(); thought_col = None

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with chat_col:
                if msg["role"] == "user":
                    with st.chat_message("You"): st.markdown(msg['content'])
                else:
                    with st.chat_message("Marin"): 
                        if "[[IMG_PATH:" in msg['content']:
                            txt = msg['content'].split("[[IMG_PATH:")[0]
                            img = msg['content'].split("[[IMG_PATH:")[1].split("]]")[0]
                            st.markdown(txt)
                            if os.path.exists(img): st.image(img)
                        else: st.markdown(msg['content'])
            if thought_col and "thought" in msg and msg["thought"]:
                with thought_col:
                    with st.chat_message("assistant", avatar="🧠"): st.info(msg["thought"])

    if prompt := st.chat_input("Speak...", disabled=st.session_state.processing):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.processing = True
        st.rerun()

    if st.session_state.processing:
        with st.spinner(st.session_state.settings["status_text"]):
            # 1. Analyze & Brain Work
            inputs = extract_sensory_data(st.session_state.messages[-1]['content'])
            st.session_state.last_inputs = inputs 
            complexity = torch.mean(inputs).item()
            st.session_state.density += (1.0 + (complexity * 2.0))
            if st.session_state.density > (st.session_state.brain.hidden_size * 4): 
                msg = st.session_state.brain.grow_brain()
                st.session_state.notifications.append(msg)
                st.session_state.optimizer = optim.Adam(st.session_state.brain.parameters(), lr=st.session_state.settings["learning_rate"])
            st.session_state.optimizer.zero_grad()
            pred = st.session_state.brain(inputs)
            target = pred.clone().detach()
            if inputs[1] > 0.5: target[2] = 0.95; target[5] = 0.2 
            if inputs[2] > 0.5: target[4] = 0.8; target[5] = 0.1  
            if inputs[5] > 0.5: target[3] = 0.9; target[5] = 1.0  
            loss = torch.mean((pred - target)**2)
            loss.backward(); st.session_state.optimizer.step()
            st.session_state.traits = pred.detach().numpy()
            
            # 5. GENERATE RESPONSE
            p_txt, d_txt = load_text_files()
            t = st.session_state.traits
            len_instr = "Keep responses SHORT (2 sentences). No monologues." if st.session_state.settings["response_len"] == "Short & Punchy" else "Be detailed and expressive. Write longer paragraphs."
            img_instr = "3. To generate an image, write: [IMG: description]." if st.session_state.settings["img_gen"] else "3. Do not generate images."
            
            system_prompt = f"""
            {p_txt}
            [SPEECH EXAMPLES]
            {d_txt}
            [Brain State: Energy={t[0]:.2f}, Openness={t[1]:.2f}, Empathy={t[2]:.2f}, Attraction={t[3]:.2f}, Anxiety={t[4]:.2f}, Mood={t[5]:.2f}]
            [User Emotion Detected: Joy={inputs[0]:.1f}, Sadness={inputs[1]:.1f}, Anger={inputs[2]:.1f}, Fear={inputs[3]:.1f}, Love={inputs[5]:.1f}]
            [INSTRUCTIONS]
            1. {len_instr}
            2. React to the 'User Emotion Detected'.
            3. START every response with your inner thoughts inside [THOUGHTS: ...].
               Example: [THOUGHTS: Oh wow, he's actually talking to me! Be cool!] "Yeah, whatever."
            {img_instr}
            """
            
            try:
                response = ""
                backend = st.session_state.settings["backend"]

                # --- OPTION A: OPENAI (NEW) ---
                if "OpenAI" in backend:
                    key = st.session_state.get("openai_key")
                    if not key:
                         st.error("Missing OpenAI Key!")
                         st.stop()
                    
                    client = OpenAI(api_key=key)
                    msgs = [{'role':'system', 'content':system_prompt}] + st.session_state.messages[-10:]
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini", # Good balance of cost/speed
                        messages=msgs
                    )
                    response = completion.choices[0].message.content

                # --- OPTION B: CLOUD API (HF) ---
                elif "HuggingFace" in backend:
                    # Try secrets, then manual
                    try: token = st.secrets["HF_TOKEN"]
                    except: token = st.session_state.get("hf_key")
                    
                    if not token:
                        st.error("Missing HuggingFace Token!")
                        st.stop()
                        
                    client = InferenceClient(token=token)
                    msgs = [{'role':'system', 'content':system_prompt}] + st.session_state.messages[-10:]
                    for t in client.chat_completion(messages=msgs, model="mistralai/Mistral-7B-Instruct-v0.3", max_tokens=250, stream=True):
                        response += t.choices[0].delta.content or ""
                
                # --- OPTION C: LOCAL OLLAMA ---
                elif "Ollama" in backend:
                    msgs = [{'role':'system', 'content':system_prompt}] + st.session_state.messages[-10:]
                    res = ollama.chat(model='llama3:8b', messages=msgs)
                    response = res['message']['content']

                # Common Parsing
                thought_match = re.search(r'\[THOUGHTS: (.*?)\]', response, re.DOTALL | re.IGNORECASE)
                thought_text = thought_match.group(1) if thought_match else None
                final = re.sub(r'\[THOUGHTS: .*?\]', '', response, flags=re.DOTALL | re.IGNORECASE).strip()
                if st.session_state.settings["img_gen"]:
                    match = re.search(r'\[IMG: (.*?)\]', final, re.IGNORECASE)
                    if match:
                        img_path = generate_image(match.group(1))
                        final = final.replace(match.group(0), "") + f"[[IMG_PATH:{img_path}]]"
                
                st.session_state.messages.append({"role": "assistant", "content": final, "thought": thought_text})
                save_system()

            except Exception as e:
                st.error(f"Generation Error: {e}")
            
        st.session_state.processing = False
        st.rerun()
