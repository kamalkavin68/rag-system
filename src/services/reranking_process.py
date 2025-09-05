from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Tuple, Set
from langchain_core.documents import Document


class RerankingProcess:
    def __init__(self):
        self.embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.nlp = spacy.load("en_core_web_sm")

    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        return list(set(
            token.lower().strip('.,:;!?()[]{}') for token in tokens if len(token) > 1
        ))

    def extract_nlp_features(self, text: str) -> dict:
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

    def _custom_score(self, sim: float, meta: dict, question_terms: Set[str]) -> float:
        meta_nouns = set(word.lower() for word in meta.get('nouns', []))
        meta_entities = set(ent.lower() for ent in meta.get('entities', []))
        meta_chunks = set(chunk.lower() for chunk in meta.get('noun_chunks', []))

        match_score = 0.0
        for term in question_terms:
            if term in meta_nouns:
                match_score += 0.05
            if term in meta_entities:
                match_score += 0.1
            if term in meta_chunks:
                match_score += 0.07

        return sim + match_score

    def rerank(self, question: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        question_embedding = self.embed_model.embed_query(question)
        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = self.embed_model.embed_documents(doc_texts)
        similarities = cosine_similarity([question_embedding], doc_embeddings)[0]

        question_meta = self.extract_nlp_features(question)
        question_terms = set(
            question_meta.get("entities", []) + question_meta.get("nouns", [])
        )

        scored_docs = [
            (doc, self._custom_score(similarities[i], doc.metadata, question_terms))
            for i, doc in enumerate(docs)
        ]

        return sorted(scored_docs, key=lambda x: x[1], reverse=True)
