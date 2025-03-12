# limit the numpy threads to improve performance
import os
import multiprocessing
os.environ["OMP_NUM_THREADS"] = str(int(multiprocessing.cpu_count()) // 2)


# now we continue
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import memryx as mx
import argparse
from time import sleep

import streamlit as st

def top_k_indices(arr, k):
    indices = np.argpartition(arr, -k)[-k:]
    sorted_indices = indices[np.argsort(-arr[indices])]
    return arr[sorted_indices], sorted_indices

class MxaTinyStories:
    """
    MxaTinyStories class to load and run the tiny stories 33M parameter model that generates multiple tiny stories
    based on a single prompt.
    Parameters
    ----------
    data_dir: string
        Absolute path to the directory containing the DFP and embedding/reverse-embedding matrices
    max_len: int
        Context length of the model. This is defined while exporting the model to onnx.
    """
    def __init__(self, data_dir, max_len = 128):

        self.dfp = mx.Dfp(f"{data_dir}/tinystories33M.dfp")

        self.mxa_core_model = mx.SyncAccl(self.dfp)
        self.max_len = max_len
        if max_len != 128:
            raise RuntimeError("Cannot use context lengths other than 128 with this DFP!")

        self.embedding_table = np.load(f"{data_dir}/embedding_table.npy").astype(np.float32)
        self.pos_embedding_mat = np.load(f"{data_dir}/positional_embedding_matrix.npy").astype(np.float32)[0]
        self.rev_embedding_matrix = np.load(f"{data_dir}/reverse_embedding_matrix.npy").astype(np.float32)
        self.num_tokens = 0

    def run_inference(self, input_data):
        """
        Run the inference return the output given the input tokens
        Parameters
        ----------
        input_data: List[np.array()]
            List to input token sequences of size (1x128)
        """
        core_input = []
        for inp in input_data:
            core_inp = np.add( self.embedding_table[inp[0]], self.pos_embedding_mat )
            core_inp = core_inp.reshape(self.dfp.input_shapes[0]).astype(np.float32)
            core_input.append(core_inp)
            channels = core_input[0].shape[-1]
        core_output = self.mxa_core_model.run(core_input)
        rev_input = []
        if isinstance(core_output,list):
            for out in core_output:
                #converting the channel last output back to channel first for onnx post processing
                out = out.reshape([self.max_len,channels])
                rev_input.append(out)
        else:
            rev_input = [core_output.reshape([self.max_len,channels])]
        output_data = []
        for out in rev_input:
            output_data.append(np.matmul(out,self.rev_embedding_matrix))
        return output_data

    def generate(self,tokenizer, prompt, beam_width=1):
        """
        Generate the output token and print the latest token that eventually generates a full story
        Parameters
        ----------
        tokenizer: transformers.tokenizer()
            Official tokenizer for this model
        prompt: np.array()
            Tokenized input prompt
        beam_width: int
            Length of beam search(defaults to 3). Larger beam_width might give better results but takes longer to compute
        """
        init_len = len(prompt)
        sequences = [prompt]
        scores = [0]
        story_output = tokenizer.decode(prompt, skip_special_tokens=True)  # Start with the initial decoded prompt

        # Placeholder for displaying the story as it generates
        story_placeholder = st.empty()

        try:
            while(True):
                all_candidates = []

                n = len(sequences)
                # fill with dummy tokens
                inp = [np.full([1,self.max_len], 50256, dtype=np.int64) for _ in range(beam_width)]
                idx = 0
                for i in range(n):
                    idx = len(sequences[i])
                    inp[i][:,0:idx] = sequences[i]
                out = self.run_inference(inp)
                for i in range(n):
                    seq = sequences[i]
                    score = scores[i]
                    y = out[i][idx-1,:]
                    top_k_probs, top_k_tokens = top_k_indices(y, beam_width)
                    for j in range(beam_width):
                        if(len(seq)==self.max_len):
                            seq.pop(0)
                        candidate_seq = seq + [top_k_tokens[j]]
                        candidate_score = score + top_k_probs[j]
                        all_candidates.append((candidate_seq, candidate_score))
                all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                sequences, scores = zip(*all_candidates[:beam_width])

                 # Decode the latest token and append it to the story
                token_text = tokenizer.decode([sequences[0][-1]], skip_special_tokens=True)
                story_output += token_text  # Append the token to the ongoing story text
                story_placeholder.write(story_output)  # Update the placeholder with the latest story

                if all(seq[-1] == tokenizer.eos_token_id for seq in sequences):
                    self.num_tokens += 1
                    break
                else:
                    self.num_tokens += 1
        except KeyboardInterrupt:
            st.write("Generation interrupted.")

        print("")

# Initialize Streamlit app layout
st.title("Tiny Story Generator")
st.write("Enter the beginning of a story and let the AI generate the rest!")

# User input for the story prompt
prompt = st.text_input("Type the beginning of a story:", "")

# Load tokenizer and model
@st.cache_resource
def load_model(data_dir, max_len=128):
    model = MxaTinyStories(data_dir, max_len)
    tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories-33M')
    return model, tokenizer

# Specify data directory
data_dir = "../models" if os.path.exists("../models") else "."
model, tokenizer = load_model(data_dir)

# Beam width input
beam_width = st.slider("Beam Width", min_value=1, max_value=2, value=1)

# Generate button only enabled if the prompt is not empty
if prompt:
    if st.button("Generate Story"):
        # Encode the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="np")

        # Start generating the story
        st.write("**Story Continuation:**")
        story = model.generate(tokenizer, input_ids.tolist()[0], beam_width)
        
else:
    st.warning("Please enter the beginning of your story to generate a continuation.")
