import streamlit as st

def rag_general():
    # st.image("./images/RAG.png")
    st.markdown("""
                
                RAG (Retrieval-Augmented Generation) is a technique used to improve the performance of language models like LLMs (Large Language Models) by incorporating your specific data. With RAG, your data is added to the existing pool of data that LLMs are trained on. This allows the model to provide more relevant responses to user queries.
                
                
                
                """)
    # st.info("The Exchange Of Methods And Algorithms Between Human And Machine To Deep Learn And Apply Problem Solving Is Known As Deep Learning (DL) ― P.S. Jagadeesh Kumar")


def rag_evaluation():
    st.header("🧠Retrieval Evaluation 👩‍🏫 Generation Evaluation")
    st.markdown("""

    Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. RAG denotes a class of LLM applications that use external data to augment the LLM’s context. There are existing tools and frameworks that help you build these pipelines but evaluating it and quantifying your pipeline performance can be hard. This is where Ragas (RAG Assessment) comes in.    
    """)
    # st.image("./images/lstm.png")
    st.markdown("""
#### Accuracy: 
This is like checking how often your solution gets the right answer. For example, if you’re solving math problems, accuracy measures how many correct answers you get compared to the total number of problems you solve.
#### Faithfulness: 
 This is about how closely your solution matches what’s expected or true. For instance, if you’re summarizing a story, faithfulness checks if your summary captures all the important points accurately.
#### Speed: 
 This measures how quickly your solution works. If you’re trying to solve a maze, speed evaluates how fast you find the way out.

    """)


def rag_loading():
    # st.image("./images/RAG.png")
    st.markdown("""
                
    1. Nodes and Documents: Imagine each of your papers as a piece of data. A “Document” is like a big container where we put all these papers together. This container could be like a big box where you keep all your papers. Now, a “Node” is like one paper from that big box. Each paper(or Node) has some information attached to it, like where it came from or how it relates to other papers.
    2. Connectors: These are like special tools we use to pick up papers from different places and put them into our big box. Just like how you might use your hand to pick up papers from under your bed, these connectors help us gather data from different sources like files, websites, or databases and organize them into Documents and Nodes.                
                
                
                """)
    # st.info("The Exchange Of Methods And Algorithms Between Human And Machine To Deep Learn And Apply Problem Solving Is Known As Deep Learning (DL) ― P.S. Jagadeesh Kumar")

