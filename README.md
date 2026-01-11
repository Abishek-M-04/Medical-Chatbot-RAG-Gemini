# Medical-Chatbot-RAG-Gemini
A simple medical chatbot using RAG and Gemini API for question-answering
Medical Chatbot (RAG + Gemini)
Overview

This project is a medical chatbot designed to answer medical-related questions using a Retrieval-Augmented Generation (RAG) approach and the Gemini API for response generation. It is based on a small medical text file, and the system retrieves relevant information from it to provide accurate answers to user queries.

What we will use:

Python: Programming language for building the chatbot.

Gemini API: Cloud-based API to generate responses based on the input data.

FAISS (CPU): For performing fast similarity searches to retrieve relevant information from the medical text.

Simple CLI: A command-line interface (CLI) for interaction, to keep the setup lightweight and efficient.

What we will NOT use:

No CUDA: No GPU acceleration is required.

No Docker: The chatbot runs without Docker to save resources.

No HuggingFace local models or Pinecone: We focus on simplicity and lightweight operation.
