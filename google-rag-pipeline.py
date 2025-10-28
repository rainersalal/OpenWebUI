"""
title: Google Shared Drive RAG Filter (Recursive)
author: Custom
date: 2025-10-28
version: 1.5
license: MIT
description: A filter pipeline for retrieving and querying all documents from a Google Workspace Shared Drive recursively
requirements: google-auth, google-auth-oauthlib, google-auth-httplib2, google-api-python-client, llama-index-core, llama-index-readers-google, llama-index-embeddings-huggingface, sentence-transformers
"""

from typing import List, Optional
import os
import json
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        GOOGLE_DRIVE_CREDENTIALS: str = ""
        SHARED_DRIVE_ID: str = "0AP1X1awOBR73Uk9PVA"
        OPENAI_API_KEY: Optional[str] = None
        EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
        TOP_K: int = 5
        CHUNK_SIZE: int = 1024
        ENABLE_RAG: bool = True
        AUTO_REINDEX: bool = False

    def __init__(self):
        self.type = "filter"
        self.name = "Google Shared Drive RAG (Recursive)"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
            }
        )
        self.index = None
        self.retriever = None
        self.documents = []
        self.initialized = False

    async def on_startup(self):
        """Initialize the pipeline on startup"""
        print("🚀 Starting Google Drive RAG Filter (Recursive)...")

        if not self.valves.GOOGLE_DRIVE_CREDENTIALS:
            print("⚠️  No Google Drive credentials provided.")
            return

        if not self.valves.SHARED_DRIVE_ID:
            print("⚠️  No Shared Drive ID provided.")
            return

        try:
            await self._initialize_index()
            self.initialized = True
        except Exception as e:
            print(f"❌ Error during startup: {str(e)}")
            import traceback
            traceback.print_exc()

    async def on_shutdown(self):
        """Cleanup on shutdown"""
        print("👋 Shutting down Google Drive RAG Filter...")

    async def on_valves_updated(self):
        """Called when valves are updated in the UI"""
        print("🔄 Valves updated, reinitializing index...")
        if self.valves.GOOGLE_DRIVE_CREDENTIALS and self.valves.SHARED_DRIVE_ID:
            try:
                await self._initialize_index()
                self.initialized = True
            except Exception as e:
                print(f"❌ Error reinitializing: {str(e)}")

    async def _initialize_index(self):
        """Initialize or re-initialize the document index"""
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.readers.google import GoogleDriveReader

        # Parse credentials
        try:
            credentials_dict = json.loads(self.valves.GOOGLE_DRIVE_CREDENTIALS)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in credentials: {str(e)}")
            return

        # Save credentials to temporary file
        creds_path = "/tmp/google_credentials.json"
        with open(creds_path, 'w') as f:
            json.dump(credentials_dict, f)

        # Set up embedding model
        has_openai_key = (
            self.valves.OPENAI_API_KEY
            and self.valves.OPENAI_API_KEY.strip() != ""
            and self.valves.OPENAI_API_KEY.strip().lower() not in ["-", "none", "n/a"]
        )

        if has_openai_key:
            print("🔑 Using OpenAI embeddings")
            os.environ["OPENAI_API_KEY"] = self.valves.OPENAI_API_KEY
            from llama_index.embeddings.openai import OpenAIEmbedding
            Settings.embed_model = OpenAIEmbedding()
        else:
            print(f"🤖 Using local embeddings: {self.valves.EMBEDDING_MODEL}")
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.valves.EMBEDDING_MODEL
            )

        # Configure chunk settings
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.valves.CHUNK_SIZE,
            chunk_overlap=200
        )

        # No LLM needed for retrieval
        Settings.llm = None

        # Initialize Google Drive Reader
        loader = GoogleDriveReader(
            service_account_key_path=creds_path
        )

        # Load all documents recursively from the Shared Drive
        print(f"📂 Loading all documents from Shared Drive ID: {self.valves.SHARED_DRIVE_ID}")
        self.documents = loader.load_data(drive_id=self.valves.SHARED_DRIVE_ID)
        print(f"✅ Loaded {len(self.documents)} documents")

        # Create index
        if self.documents:
            print("🔨 Creating vector index...")
            self.index = VectorStoreIndex.from_documents(self.documents)
            self.retriever = self.index.as_retriever(
                similarity_top_k=self.valves.TOP_K
            )
            print(f"✅ Index created successfully with {len(self.documents)} documents")
        else:
            print("⚠️  No documents found in Shared Drive")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process incoming requests - called BEFORE the LLM"""
        if not self.valves.ENABLE_RAG:
            return body

        if self.valves.AUTO_REINDEX and self.initialized:
            print("🔄 Auto-reindexing...")
            await self._initialize_index()

        if not self.retriever:
            return body

        try:
            messages = body.get("messages", [])
            if not messages:
                return body

            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

            if not user_message:
                return body

            # Retrieve relevant documents
            retrieved_nodes = self.retriever.retrieve(user_message)
            if not retrieved_nodes:
                print("ℹ️  No relevant documents found")
                return body

            context_parts = []
            sources = []
            for i, node in enumerate(retrieved_nodes, 1):
                content = node.node.get_content()
                metadata = node.node.metadata
                file_name = metadata.get('file_name', 'Unknown')
                context_parts.append(f"[Document {i}: {file_name}]\n{content}\n")
                sources.append(file_name)

            context = "\n".join(context_parts)

            augmented_content = f"""Based on the following documents from our Google Shared Drive, please answer the user's question.

**Retrieved Documents:**
{context}

**User Question:** {user_message}

**Instructions:** 
- Answer based on the information in the documents above
- If the documents don't contain relevant information, say so clearly
- Cite which documents you used
- Be accurate and concise"""

            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i]["content"] = augmented_content
                    break

            body["messages"] = messages
            print(f"✅ Added context from {len(sources)} documents: {', '.join(sources)}")

        except Exception as e:
            print(f"❌ Error in inlet: {str(e)}")
            import traceback
            traceback.print_exc()

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process outgoing responses - called AFTER the LLM"""
        return body
