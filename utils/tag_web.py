import streamlit as st, json, csv, pandas as pd, os
from streamlit_extras.add_vertical_space import add_vertical_space
from ragas import evaluate
from datasets import Dataset 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ragas.metrics.critique import harmfulness
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness


def news():
    nextstep = False
    st.write('Follow the pipeline: 🆕Retrieve ➡️ Chunking')
    results = []
    contexts = []
    queries = [
        "Who discovered the Galapagos Islands and how?",
        "What is Brooklyn–Battery Tunnel?",
        "Are Penguins found in the Galapagos Islands?",
        "What is the significance of the Statue of Liberty in New York City?",]

    ground_truths = [
        "The Galapagos Islands were discovered in 1535 by the bishop of Panama, Tomás de Berlanga, whose ship had drifted off course while en route to Peru. He named them Las Encantadas (“The Enchanted”), and in his writings he marveled at the thousands of large galápagos (tortoises) found there. Numerous Spanish voyagers stopped at the islands from the 16th century, and the Galapagos also came to be used by pirates and by whale and seal hunters. ",
        "The Brooklyn-Battery Tunnel (officially known as the Hugh L. Carey Tunnel) is the longest continuous underwater vehicular tunnel in North America and runs underneath Battery Park, connecting the Financial District in Lower Manhattan to Red Hook in Brooklyn.[586]",
        "Penguins live on the galapagos islands side by side with tropical animals.",
        "The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th and early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an iconic landmark and a global symbol of cultural diversity and freedom.",]

    if not os.path.exists('./data/news_contexts.csv'):
        if st.button('Retrieve News'):
            st.text(" creating ./data/news_contexts.csv.")
            urls = [
                "https://en.wikipedia.org/wiki/New_York_City",
                "https://www.britannica.com/place/Galapagos-Islands",
            ]        
            
            with st.spinner('Scraping news ...'):
                # collect data using selenium url loader
                loader = SeleniumURLLoader(urls=urls)
                documents = loader.load()
                documentList = []
                for doc in documents:
                    d = str(doc.page_content).replace("\\n", " ").replace("\\t"," ").replace("\n", " ").replace("\t", " ")
                    documentList.append(d)
            st.text(documentList[:1])
            
            with st.spinner("embedding, chunking ..."):
                st.write(11)
                embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                st.write(21)
                text_splitter = SemanticChunker(embedding_function)
                st.write(31)
                docs = text_splitter.create_documents(documentList)
                # storing embeddings in a folder
                st.write(123)                
                vector_store = Chroma.from_documents(docs, embedding_function, persist_directory="./data/chroma_db_news")
                # use this to load vector database
                st.write(134)
                vector_store = Chroma(persist_directory="./data/chroma_db_news", embedding_function=embedding_function)
                
                st.write('Great!')
                PROMPT_TEMPLATE = """
                    Go through the context and answer given question strictly based on context. 
                    Context: {context}
                    Question: {question}
                    Answer:
                    """
                    
                qa_chain = RetrievalQA.from_chain_type(
                        llm = ChatOpenAI(temperature=0),
                        # retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
                        retriever=vector_store.as_retriever(),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)})
                
                for query in queries:
                    st.text(query)
                    result = qa_chain({"query": query})
                    st.text(result)
                
                    results.append(result['result'])
                    sources = result["source_documents"]
                    contents = []
                    for i in range(len(sources)):
                        contents.append(sources[i].page_content)
                    contexts.append(contents)
                    
                df = pd.DataFrame(contexts)
                df_results = pd.DataFrame(results)
                df.to_csv('./data/news_contexts.csv', index=False)  # Save without index
                df.to_json("./data/news_contexts.json", orient="records")  # Save each row as a
                df_results.to_csv('./data/news_results.csv', index=False)  # Save without index
                df_results.to_json("./data/news_results.json", orient="records")  # Save each row as a
                
                if 'news' not in st.session_state:   st.session_state['news'] = ''
                st.session_state['news'] = contexts
    else:
        contexts = pd.read_json('./data/news.json')
        results  = pd.read_json('./data/news.json')
        st.text(contexts)
        
        
        st.code("""
        
        d = {
            "question": queries,
            "answer": results,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        dataset = Dataset.from_dict(d)
        score = evaluate(dataset,metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness, harmfulness])
        score_df = score.to_pandas()
        score_df.to_csv("./data/EvaluationScores.csv", encoding="utf-8", index=False)
        st.text(score_df)
        """)
        
        st.code("""
                 
                 faithfulness             0.955000
answer_relevancy         0.923192
context_precision        0.733333
context_recall           0.916667
context_entity_recall    0.322197
answer_similarity        0.941792
answer_correctness       0.665889
harmfulness              0.000000
dtype: float64
                 
                 """)