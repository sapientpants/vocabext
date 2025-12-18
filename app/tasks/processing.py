"""Document processing pipeline."""

import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session
from app.models import Document, Word, Extraction
from app.services.extractor import TextExtractor
from app.services.tokenizer import Tokenizer
from app.services.enricher import Enricher

logger = logging.getLogger(__name__)


async def process_document(document_id: int) -> None:
    """
    Process a document through the full extraction pipeline.

    Steps:
    1. Extract text from the file
    2. Tokenize and lemmatize
    3. Check for duplicates against existing vocabulary
    4. Enrich new words via LLM
    5. Update document status
    """
    async with async_session() as session:
        try:
            # Get document
            document = await session.get(Document, document_id)
            if not document:
                logger.error(f"Document {document_id} not found")
                return

            logger.info(f"Processing document: {document.filename}")

            # Step 1: Extract text
            file_path = settings.upload_dir / document.filename
            extractor = TextExtractor(whisper_model=settings.whisper_model)

            try:
                raw_text = await extractor.extract(file_path)
                document.raw_text = raw_text
                await session.commit()
            except Exception as e:
                logger.error(f"Text extraction failed: {e}")
                document.status = "error"
                document.error_message = f"Text extraction failed: {e}"
                await session.commit()
                return

            # Step 2: Tokenize
            tokenizer = Tokenizer(model_name=settings.spacy_model)
            tokens = tokenizer.tokenize(raw_text)

            if not tokens:
                logger.warning(f"No tokens extracted from document {document_id}")
                document.status = "pending_review"
                await session.commit()
                return

            # Step 3: Check for duplicates and create extractions
            enricher = Enricher()

            for token in tokens:
                # Check if word already exists in vocabulary
                existing_word = await _find_existing_word(
                    session, token.lemma, token.pos
                )

                if existing_word:
                    # Create duplicate extraction linked to existing word
                    extraction = Extraction(
                        document_id=document_id,
                        word_id=existing_word.id,
                        surface_form=token.surface_form,
                        lemma=token.lemma,
                        pos=token.pos,
                        gender=existing_word.gender,
                        plural=existing_word.plural,
                        preterite=existing_word.preterite,
                        past_participle=existing_word.past_participle,
                        auxiliary=existing_word.auxiliary,
                        translations=existing_word.translations,
                        context_sentence=token.context_sentence,
                        status="duplicate",
                    )
                    session.add(extraction)
                else:
                    # Step 4: Enrich new word via LLM
                    try:
                        enrichment = await enricher.enrich(
                            token.lemma, token.pos, token.context_sentence
                        )
                    except Exception as e:
                        logger.warning(f"Enrichment failed for '{token.lemma}': {e}")
                        enrichment = None

                    extraction = Extraction(
                        document_id=document_id,
                        surface_form=token.surface_form,
                        lemma=token.lemma,
                        pos=token.pos,
                        gender=enrichment.gender if enrichment else None,
                        plural=enrichment.plural if enrichment else None,
                        preterite=enrichment.preterite if enrichment else None,
                        past_participle=enrichment.past_participle if enrichment else None,
                        auxiliary=enrichment.auxiliary if enrichment else None,
                        translations=json.dumps(enrichment.translations) if enrichment else None,
                        context_sentence=token.context_sentence,
                        status="pending",
                    )
                    session.add(extraction)

                # Commit periodically to avoid memory issues
                await session.commit()

            # Step 5: Update document status
            document.status = "pending_review"
            await session.commit()

            logger.info(f"Document {document_id} processing complete")

        except Exception as e:
            logger.exception(f"Processing failed for document {document_id}: {e}")
            try:
                document = await session.get(Document, document_id)
                if document:
                    document.status = "error"
                    document.error_message = str(e)
                    await session.commit()
            except Exception:
                pass


async def _find_existing_word(
    session: AsyncSession, lemma: str, pos: str
) -> Word | None:
    """
    Find an existing word in vocabulary.

    For nouns, we match on lemma+pos (gender checked separately).
    For other POS, we match on lemma+pos.
    """
    stmt = select(Word).where(Word.lemma == lemma, Word.pos == pos)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
