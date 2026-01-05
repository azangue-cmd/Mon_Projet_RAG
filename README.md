               Local RAG - Secure Document Assistant

Description : Application développée en Python permettant d'interroger des documents sensibles (PDF) via une Intelligence Artificielle Générative.

Points forts Cybersécurité & Tech :

Confidentialité totale (Data Privacy) : Utilisation d'un LLM local (Ollama/Mistral), aucune donnée n'est envoyée dans le cloud (contrairement à ChatGPT).

RAG Architecture : Implémentation d'un pipeline complet (Ingestion -> Chunking -> Vector Store -> Retrieval -> Generation).

Chat History : Tu utilises st.session_state pour garder le fil de la conversation (comme ChatGPT).

Source Retrieval : Le modèle n'invente pas, il montre les extraits exacts du PDF utilisés.

Nettoyage de session : Un bouton pour vider la mémoire et changer de fichier proprement.

Stack : LangChain, ChromaDB, Streamlit, Python.

Utilisation : Idéal pour l'analyse de rapports d'audit ou de documentations techniques en environnement sécurisé (Air-gapped).