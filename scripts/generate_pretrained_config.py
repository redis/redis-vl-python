#!/usr/bin/env python3
"""Generate pretrained LLM Router configurations with pre-computed embeddings.

This script uses HFTextVectorizer to embed reference phrases and serialize
a PretrainedRouterConfig to JSON. Run once to regenerate default.json.

Usage:
    python scripts/generate_pretrained_config.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from redisvl.extensions.llm_router.schema import (
    PretrainedReference,
    PretrainedRouterConfig,
    PretrainedTier,
)
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

# Three tiers grounded in Bloom's Taxonomy + NVIDIA LLM Router Blueprint categories
TIERS = [
    {
        "name": "simple",
        "model": "openai/gpt-4.1-nano",
        "distance_threshold": 0.5,
        "metadata": {
            "provider": "openai",
            "cost_per_1k_input": 0.0001,
            "cost_per_1k_output": 0.0004,
            "blooms_taxonomy": ["Remember", "Understand"],
        },
        "references": [
            "hello how are you",
            "what is the capital of France",
            "convert this to JSON",
            "is this email spam or not spam",
            "what time is it",
            "thanks for your help",
            "translate hello to Spanish",
            "what does FAQ stand for",
            "list the days of the week",
            "goodbye see you later",
            "what is two plus two",
            "define the word algorithm",
            "how do you spell necessary",
            "what color is the sky",
            "who wrote Romeo and Juliet",
            "what year did World War 2 end",
            "how many continents are there",
            "what is the boiling point of water",
        ],
    },
    {
        "name": "standard",
        "model": "anthropic/claude-sonnet-4-5",
        "distance_threshold": 0.6,
        "metadata": {
            "provider": "anthropic",
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
            "blooms_taxonomy": ["Apply", "Analyze"],
        },
        "references": [
            "explain how neural networks learn",
            "summarize this research paper in detail",
            "write a blog post about climate change",
            "debug this Python function and explain the issue",
            "compare and contrast these two approaches",
            "analyze the sentiment of this customer feedback",
            "write unit tests for this class",
            "explain the tradeoffs between SQL and NoSQL databases",
            "create a marketing email for our new product",
            "review this pull request for potential issues",
            "explain how garbage collection works in Java",
            "write a detailed product description for",
            "help me refactor this code for better readability",
            "outline the pros and cons of microservices architecture",
            "draft a project proposal for a new feature",
            "explain the difference between TCP and UDP protocols",
            "write a recursive function to traverse a binary tree",
            "analyze this dataset and identify key trends",
        ],
    },
    {
        "name": "expert",
        "model": "anthropic/claude-opus-4-5",
        "distance_threshold": 0.7,
        "metadata": {
            "provider": "anthropic",
            "cost_per_1k_input": 0.005,
            "cost_per_1k_output": 0.025,
            "blooms_taxonomy": ["Evaluate", "Create"],
        },
        "references": [
            "prove this mathematical theorem step by step",
            "architect a distributed system for millions of concurrent users",
            "write a research paper analyzing the impact of",
            "design a novel algorithm for graph partitioning",
            "review this legal contract and identify potential issues",
            "develop a comprehensive business strategy for market entry",
            "create a detailed technical specification for a new platform",
            "write a compiler optimization pass for loop unrolling",
            "formulate a hypothesis and design an experiment to test it",
            "analyze the philosophical implications of artificial general intelligence",
            "design a machine learning pipeline for production deployment",
            "write a formal verification proof for this consensus protocol",
            "create a comprehensive security audit report for this system",
            "develop a novel approach to solving this NP-hard problem",
            "write a peer-reviewed analysis of quantum computing advances",
            "architect a fault-tolerant database replication system",
            "design an operating system scheduler with real-time guarantees",
            "create a mathematical model for predicting market volatility",
        ],
    },
]

OUTPUT_PATH = (
    Path(__file__).parent.parent
    / "redisvl"
    / "extensions"
    / "llm_router"
    / "pretrained"
    / "default.json"
)


def main():
    print("Initializing HFTextVectorizer (sentence-transformers/all-mpnet-base-v2)...")
    vectorizer = HFTextVectorizer(
        model="sentence-transformers/all-mpnet-base-v2"
    )

    pretrained_tiers = []
    for tier_def in TIERS:
        print(f"Embedding {len(tier_def['references'])} references for tier '{tier_def['name']}'...")
        vectors = vectorizer.embed_many(tier_def["references"])

        references = [
            PretrainedReference(text=text, vector=vec)
            for text, vec in zip(tier_def["references"], vectors)
        ]

        pretrained_tiers.append(
            PretrainedTier(
                name=tier_def["name"],
                model=tier_def["model"],
                references=references,
                metadata=tier_def["metadata"],
                distance_threshold=tier_def["distance_threshold"],
            )
        )

    config = PretrainedRouterConfig(
        name="llm-router-default",
        version="1.0.0",
        vectorizer={
            "type": "hf",
            "model": "sentence-transformers/all-mpnet-base-v2",
        },
        tiers=pretrained_tiers,
        routing_config={
            "max_k": 1,
            "aggregation_method": "avg",
            "cost_optimization": False,
        },
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    print(f"Wrote pretrained config to {OUTPUT_PATH}")
    print(f"Total tiers: {len(pretrained_tiers)}")
    for tier in pretrained_tiers:
        print(f"  {tier.name}: {len(tier.references)} references, model={tier.model}")


if __name__ == "__main__":
    main()
