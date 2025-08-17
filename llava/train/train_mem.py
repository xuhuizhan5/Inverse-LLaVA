import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
