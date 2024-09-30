"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

Adapted from: https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/refs/heads/master/data/fineweb.py

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from datasets import load_dataset
from transformers import AutoTokenizer

num_proc = 8
model_name = "meta-llama/Llama-2-7b-hf"
model_cache_dir = "/fs/nexus-scratch/psando/huggingface"
data_cache_dir = '/fs/nexus-scratch/psando/fineweb-10b'         # where to store huggingface dataset
save_subdir = os.path.join(data_cache_dir, 'llama-fineweb-10b') # where this script will save the bin files
os.makedirs(save_subdir, exist_ok=True)

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
parser.add_argument("-v", "--version", type=str, default="10B", help="Which version of fineweb to use 10B|100B")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
assert args.version in ["10B", "100B"], "version must be one of 10B, 100B"
if args.version == "10B":
    local_dir = "fineweb10B"
    remote_name = "sample-10BT"
elif args.version == "100B":
    local_dir = "fineweb100B"
    remote_name = "sample-100BT"

# create the cache the local directory if it doesn't exist yet
TOKEN_SAVE_DIR = save_subdir

# download the dataset
print(f"Downloading to {os.environ['HF_HOME']}")
fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train", cache_dir=data_cache_dir, trust_remote_code=True)

# init the tokenizer
enc = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
eot = enc.eos_token_id # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode(doc["text"], add_special_tokens=False))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(TOKEN_SAVE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(TOKEN_SAVE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])