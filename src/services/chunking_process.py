import spacy
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class SemanticChunkerWithNLP:
    def __init__(
        self,
        embed_model=None,
        model_name: str = "models/embedding-001",
        breakpoint_type: str = "percentile",
        breakpoint_amount: int = 85,
        spacy_model: str = "en_core_web_sm"
    ):  
        
        self.embed_model = embed_model or GoogleGenerativeAIEmbeddings(model=model_name)
        self.chunker = SemanticChunker(
            self.embed_model,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount
        )
        self.nlp = spacy.load(spacy_model)

    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        return list(set(
            token.lower().strip('.,:;!?()[]{}') for token in tokens if len(token) > 1
        ))

    def _extract_nlp_features(self, text: str) -> dict:
        doc = self.nlp(text)
        return {
            "entities": self._normalize_tokens([ent.text for ent in doc.ents]),
            "entity_labels": list(set(ent.label_ for ent in doc.ents)),
            "nouns": self._normalize_tokens([t.text for t in doc if t.pos_ == "NOUN" and not t.is_stop]),
            "verbs": self._normalize_tokens([t.text for t in doc if t.pos_ == "VERB" and not t.is_stop]),
            "adjectives": self._normalize_tokens([t.text for t in doc if t.pos_ == "ADJ"]),
            "noun_chunks": self._normalize_tokens([chunk.text for chunk in doc.noun_chunks]),
            "sentences": [sent.text.strip() for sent in doc.sents],
            "word_count": len([t for t in doc if t.is_alpha])
        }

    def chunk_and_enrich(self, documents: List[Document]) -> List[Document]:
        enriched_chunks = []

        for doc in documents:
            chunks = self.chunker.create_documents([doc.page_content])

            for i, chunk in enumerate(chunks):
                enriched_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "length": len(chunk.page_content),
                    "source_id": f"{doc.metadata.get('source', 'unknown')}_p{doc.metadata.get('page', 'NA')}_c{i}"
                }
                enriched_metadata.update(self._extract_nlp_features(chunk.page_content))
                chunk.metadata = enriched_metadata
                enriched_chunks.append(chunk)

        return enriched_chunks
