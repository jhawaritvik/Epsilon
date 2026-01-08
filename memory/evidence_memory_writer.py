import logging
import uuid
from typing import Dict, Any, Optional
from .supabase_client import SupabaseManager

logger = logging.getLogger("EvidenceMemory")

class EvidenceMemoryWriter:
    def __init__(self):
        self.manager = SupabaseManager()

    def record_evidence(self,
                        run_id: str,
                        source_type: str,
                        source_url: str,
                        extracted_claim: str,
                        supporting_text: str,
                        confidence: str,
                        paper_title: Optional[str] = None,
                        section: Optional[str] = None):
        """
        Records a piece of research evidence.
        """
        if not self.manager.is_enabled:
            return

        try:
            data = {
                "run_id": str(run_id),
                "source_type": source_type,
                "source_url": source_url,
                "paper_title": paper_title,
                "section": section,
                "extracted_claim": extracted_claim,
                "supporting_text": supporting_text,
                "confidence": confidence
            }

            self.manager.client.table("research_evidence").insert(data).execute()
            logger.info(f"[EvidenceMemory] Recorded evidence from {source_url} for run {run_id}")

        except Exception as e:
            logger.error(f"[EvidenceMemory] Failed to record evidence: {e}")
