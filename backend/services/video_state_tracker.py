"""
EXPERIMENTAL FEATURE

Video State Tracker: Tracks subassembly switches during assembly.

NOTE: This is an experimental feature and is not part of the main VLM pipeline.
"""

from typing import Any, Optional, Set
from loguru import logger


class SubassemblyTracker:
    """Tracks which subassembly is currently being worked on."""

    def __init__(self):
        """Initialize subassembly tracker."""
        self.current_subassembly: Optional[str] = None
        self.subassembly_history: list[Dict[str, Any]] = []
        self.known_subassemblies: Set[str] = set()

    def detect_subassembly_switch(
        self,
        placement_description: str,
        frame_number: int
    ) -> bool:
        """
        Detect if the user switched to a different subassembly.

        This is a heuristic based on keywords in the placement description.

        Args:
            placement_description: Description of the current placement
            frame_number: Frame number

        Returns:
            True if subassembly switch detected
        """
        # Look for keywords indicating subassembly work
        subassembly_keywords = [
            "separate", "another", "different", "second", "tower",
            "subassembly", "side structure", "module"
        ]

        description_lower = placement_description.lower()

        # Check if any keyword is present
        has_keyword = any(kw in description_lower for kw in subassembly_keywords)

        if has_keyword:
            # Determine subassembly identifier from description
            # Simple heuristic: extract main noun phrase
            subassembly_id = self._extract_subassembly_id(placement_description)

            if subassembly_id != self.current_subassembly:
                logger.info(
                    f"  Subassembly switch detected at frame {frame_number}: "
                    f"{self.current_subassembly} → {subassembly_id}"
                )
                self.current_subassembly = subassembly_id
                self.known_subassemblies.add(subassembly_id)
                self.subassembly_history.append({
                    "frame_number": frame_number,
                    "subassembly": subassembly_id
                })
                return True

        return False

    def _extract_subassembly_id(self, description: str) -> str:
        """
        Extract a simple subassembly identifier from description.

        Args:
            description: Placement description

        Returns:
            Subassembly identifier (e.g., "main", "tower", "side")
        """
        description_lower = description.lower()

        # Simple keyword matching
        if "tower" in description_lower:
            return "tower"
        elif "side" in description_lower:
            return "side_structure"
        elif "base" in description_lower or "foundation" in description_lower:
            return "base"
        elif "separate" in description_lower or "another" in description_lower:
            return "secondary"
        else:
            return "main"

    def get_current_subassembly(self) -> Optional[str]:
        """Get the current active subassembly."""
        return self.current_subassembly

    def reset(self):
        """Reset tracker state."""
        self.current_subassembly = None
        self.subassembly_history.clear()
        self.known_subassemblies.clear()
