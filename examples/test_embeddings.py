#!/usr/bin/env python3
"""
Test script for embeddings using the OpenAI API client.
Run with: uv run python examples/test_embeddings.py

Tests both local SentenceTransformer and hosted embedding API.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE") or None
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("API_KEY") or None

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "A fast reddish-brown canine leaps above a sleepy hound",
    "Hello, world! This is a test of the embedding system.",
]


def main():
    if EMBEDDING_API_BASE:
        print(f"Testing hosted embeddings at: {EMBEDDING_API_BASE}")
        client_kwargs = {"base_url": EMBEDDING_API_BASE, "timeout": 30}
        if EMBEDDING_API_KEY:
            client_kwargs["api_key"] = EMBEDDING_API_KEY
        client = OpenAI(**client_kwargs)

        print(f"Model: {EMBEDDING_MODEL}")
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=TEST_TEXTS,
            )
            for i, item in enumerate(response.data):
                print(f"  Text {i+1}: embedding dimension = {len(item.embedding)}")
            print(f"  Usage: {response.usage}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        from sentence_transformers import SentenceTransformer

        print(f"Testing local SentenceTransformer: {EMBEDDING_MODEL}")
        hf_token = os.getenv("HF_TOKEN")
        model = SentenceTransformer(EMBEDDING_MODEL, token=hf_token)
        embeddings = model.encode(TEST_TEXTS, convert_to_numpy=True)
        for i, emb in enumerate(embeddings):
            print(f"  Text {i+1}: embedding dimension = {len(emb)}")


if __name__ == "__main__":
    main()