import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configuration de la page
st.set_page_config(page_title="Chat avec tes PDF (Local & Privé)", page_icon="🔒")
st.title("🔒 RAG Cybersécurité : Analyse de PDF en Local")
st.markdown("Ce projet utilise **Ollama (Mistral)** pour analyser des documents sensibles sans connexion internet.")

# 2. Fonction pour traiter le PDF
def process_pdf(uploaded_file):
    # Création d'un fichier temporaire pour que PyPDFLoader puisse le lire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Chargement du PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Découpage du texte en morceaux (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Création de la base de données vectorielle (Embeddings)
    # On utilise Ollama pour transformer le texte en vecteurs mathématiques
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Nettoyage du fichier temporaire
    os.remove(tmp_path)
    return vectorstore

# 3. Interface Utilisateur
uploaded_file = st.file_uploader("Dépose ton document PDF ici", type="pdf")

if uploaded_file is not None:
    st.success("Fichier chargé ! Traitement en cours...")
    
    # Création de la base de connaissance
    if "vectorstore" not in st.session_state:
        with st.spinner("Indexation du document... (Cela peut prendre un peu de temps sur CPU)"):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            st.success("Document indexé ! Tu peux poser tes questions.")

    # Zone de Chat
    question = st.text_input("Pose ta question sur le document :")

    if question:
        # 4. La partie RAG (Récupération + Génération)
        
        # Le modèle va chercher les morceaux pertinents dans le PDF
        retriever = st.session_state.vectorstore.as_retriever()
        
        # Le modèle de langage (LLM)
        llm = Ollama(model="mistral")
        
        # Le Prompt (Les instructions données à l'IA)
        template = """Tu es un assistant expert en cybersécurité. 
        Réponds à la question en te basant UNIQUEMENT sur le contexte fourni ci-dessous.
        Si la réponse n'est pas dans le document, dis "Je ne sais pas".
        
        Contexte : {context}
        
        Question : {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # La chaîne de traitement (Chain)
        chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Affichage de la réponse
        with st.spinner("L'IA réfléchit..."):
            response = chain.invoke(question)
            st.write("### Réponse :")
            st.write(response)