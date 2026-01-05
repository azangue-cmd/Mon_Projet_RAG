import streamlit as st
import os
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
st.set_page_config(page_title="Secure RAG - Assistant Documentaire", page_icon="🛡️", layout="wide")

# --- CSS PERSONNALISÉ (Pour faire "Pro") ---
st.markdown("""
<style>
    .chat-message {padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex}
    .chat-message.user {background-color: #2b313e}
    .chat-message.bot {background-color: #475063}
    .source-box {font-size: 0.8em; color: #aaa; border-left: 2px solid #D4AF37; padding-left: 10px; margin-top: 5px;}
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS ---

def sanitize_input(user_input):
    """Simple filtre de sécurité pour éviter les injections basiques."""
    blacklist = ["ignore all instructions", "system override", "delete database"]
    for word in blacklist:
        if word in user_input.lower():
            return False
    return True

def process_pdf(uploaded_file):
    """Ingère le PDF, le découpe et crée la base vectorielle."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Découpage intelligent (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mistral")
    # Persist_directory est optionnel mais utile pour garder la DB sur le disque
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    os.remove(tmp_path)
    return vectorstore

def get_response(query, vectorstore):
    """Génère la réponse avec contexte et sources."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Récupère les 3 meilleurs passages
    
    llm = Ollama(model="mistral")
    
    # Récupération des documents pertinents
    docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    template = """
    Tu es un analyste en cybersécurité expert. Utilise le contexte suivant pour répondre à la question.
    Si la réponse n'est pas dans le contexte, dis simplement que tu ne sais pas.
    Sois précis et concis.
    
    Contexte : {context}
    
    Question : {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"context": context_text, "question": query})
    return response, docs

# --- INTERFACE UTILISATEUR ---

# Sidebar (Menu latéral)
with st.sidebar:
    st.header("📂 Gestion des Documents")
    uploaded_file = st.file_uploader("Upload ton PDF (Audit, Cours, Rapport)", type="pdf")
    
    if st.button("🧹 Effacer la mémoire & Reset"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔒 Privacy Mode")
    st.info("Modèle : Mistral (Local)\nDonnées : Non partagées")

# Titre Principal
st.title("🛡️ Secure RAG : Analyseur de Documents")

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialisation du vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Traitement du fichier au chargement
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("🔒 Chiffrement et indexation du document en cours..."):
        st.session_state.vectorstore = process_pdf(uploaded_file)
        st.success("Document sécurisé et chargé en mémoire !")

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 Voir les sources utilisées"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1} (Page {source.metadata.get('page', '?')})** :")
                    st.markdown(f"_{source.page_content[:200]}..._")

# Zone de saisie utilisateur
if prompt := st.chat_input("Pose ta question sur le document..."):
    
    # 1. Vérification Sécurité (Sanitization)
    if not sanitize_input(prompt):
        st.error("⚠️ ALERTE SÉCURITÉ : Tentative d'injection détectée.")
    elif st.session_state.vectorstore is None:
        st.warning("Merci de charger un document PDF d'abord.")
    else:
        # 2. Affichage immédiat du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("Analyse des vecteurs..."):
                response_text, sources = get_response(prompt, st.session_state.vectorstore)
                
                st.markdown(response_text)
                
                # Affichage des sources dans un menu déroulant
                with st.expander("🔍 Voir les sources (Preuve)"):
                    for idx, source in enumerate(sources):
                        st.markdown(f"**Source {idx+1} (Page {source.metadata.get('page', '?')})** :")
                        st.markdown(f"_{source.page_content[:200]}..._")

        # 4. Sauvegarde dans l'historique
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "sources": sources
        })