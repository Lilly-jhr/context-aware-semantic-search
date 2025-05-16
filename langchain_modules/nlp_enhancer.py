import sys
import os
import spacy
from typing import List, Tuple, Dict
import logging
import traceback


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

NLP_MODEL_NAME = "en_core_web_sm"
try:
    logger.info(f"Loading spaCy NLP model: {NLP_MODEL_NAME}...")
    nlp = spacy.load(NLP_MODEL_NAME)
    logger.info(f"spaCy NLP model '{NLP_MODEL_NAME}' loaded successfully.")
except OSError:
    logger.error(f"spaCy model '{NLP_MODEL_NAME}' not found. ")
    logger.error(f"Please download it by running: python -m spacy download {NLP_MODEL_NAME}")
except Exception as e:
    logger.error(f"An unexpected error occurred while loading spaCy model '{NLP_MODEL_NAME}': {e}")
    logger.error(traceback.format_exc())
    nlp = None


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts named entities from a given text using spaCy.

    Args:
        text: The input string from which to extract entities.

    Returns:
        A dictionary where keys are entity labels (e.g., "PERSON", "ORG")
        and values are lists of unique entity texts found for that label.
        Returns an empty dictionary if NLP model is not loaded or text is empty.
    """
    if nlp is None:
        logger.warning("spaCy NLP model is not loaded. Cannot extract entities.")
        return {}
    if not text or not text.strip():
        logger.debug("Input text for entity extraction is empty.")
        return {}

    doc = nlp(text)
    entities: Dict[str, List[str]] = {}
    
    for ent in doc.ents:
        label = ent.label_
        ent_text = ent.text.strip()
        if ent_text: 
            if label not in entities:
                entities[label] = []
            if ent_text not in entities[label]: 
                entities[label].append(ent_text)
    
    if entities:
        logger.debug(f"Extracted entities: {entities}")
    else:
        logger.debug("No entities found in the provided text.")
        
    return entities

if __name__ == "__main__":
    logger.info(f"--- Running {__file__} directly for testing NER ---")

    if nlp is None:
        logger.error("Exiting test as spaCy model failed to load.")
        sys.exit(1)

    sample_text_1 = "Apple Inc. is looking at buying U.K. startup for $1 billion. Tim Cook announced this yesterday in London."
    sample_text_2 = "Sherry Turkle wrote 'Alone Together'. She discussed technology with Jacques Lacan in Paris."
    sample_text_3 = "This text has no obvious named entities."
    
    texts_to_test = [sample_text_1, sample_text_2, sample_text_3]

    for i, text in enumerate(texts_to_test):
        logger.info(f"\n--- Processing Sample Text {i+1} ---")
        logger.info(f"Text: \"{text}\"")
        extracted = extract_entities(text)
        if extracted:
            logger.info("Entities Found:")
            for label, ents in extracted.items():
                logger.info(f"  {label}: {', '.join(ents)}")
        else:
            logger.info("No entities found.")
            
    logger.info(f"--- Finished direct execution of {__file__} ---")