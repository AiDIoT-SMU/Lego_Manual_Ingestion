"""
Video State Tracker: Tracks assembly state and detects duplicates.

Uses perceptual hashing to detect when two frames show the same assembly state.
"""

import imagehash
from PIL import Image as PILImage
from pathlib import Path
from typing import Dict, Any, Optional, Set
from loguru import logger


class AssemblyStateTracker:
    """Tracks assembly state across frames to detect duplicates."""

    def __init__(self, hash_size: int = 16, similarity_threshold: int = 5):
        """
        Initialize assembly state tracker.

        Args:
            hash_size: Size of perceptual hash (larger = more precise)
            similarity_threshold: Max hamming distance to consider frames similar
                                (0 = identical, higher = more tolerance)
        """
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        self.seen_states: Dict[str, Dict[str, Any]] = {}
        self.placement_history: list[Dict[str, Any]] = []

    def compute_state_hash(self, frame_path: Path) -> str:
        """
        Compute perceptual hash of a frame to represent assembly state.

        Args:
            frame_path: Path to frame image

        Returns:
            Hex string representing the perceptual hash
        """
        try:
            img = PILImage.open(frame_path)
            # Use average hash (fast and effective for assembly state)
            phash = imagehash.average_hash(img, hash_size=self.hash_size)
            return str(phash)
        except Exception as e:
            logger.warning(f"Could not hash frame {frame_path}: {e}")
            return "0" * (self.hash_size * self.hash_size // 4)  # Fallback hash

    def is_duplicate_state(
        self,
        current_hash: str,
        current_frame_number: int
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if current frame shows a duplicate assembly state.

        Args:
            current_hash: Perceptual hash of current frame
            current_frame_number: Frame number

        Returns:
            Tuple of (is_duplicate, original_placement_data)
        """
        current_hash_obj = imagehash.hex_to_hash(current_hash)

        for seen_hash_str, placement_data in self.seen_states.items():
            seen_hash_obj = imagehash.hex_to_hash(seen_hash_str)
            distance = current_hash_obj - seen_hash_obj

            if distance <= self.similarity_threshold:
                logger.info(
                    f"  Frame {current_frame_number} is duplicate of frame "
                    f"{placement_data['frame_number']} (distance={distance})"
                )
                return True, placement_data

        return False, None

    def register_placement(
        self,
        frame_path: Path,
        frame_number: int,
        placement_data: Dict[str, Any]
    ):
        """
        Register a new placement frame and its assembly state.

        Args:
            frame_path: Path to placement frame
            frame_number: Frame number
            placement_data: Data about this placement (parts, action, etc.)
        """
        state_hash = self.compute_state_hash(frame_path)

        # Store in history
        entry = {
            "frame_number": frame_number,
            "state_hash": state_hash,
            **placement_data
        }
        self.placement_history.append(entry)

        # Store in seen states
        self.seen_states[state_hash] = entry

    def check_and_register(
        self,
        frame_path: Path,
        frame_number: int,
        placement_data: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if frame is duplicate, and register if not.

        Args:
            frame_path: Path to frame
            frame_number: Frame number
            placement_data: Optional placement data (if this is a new placement)

        Returns:
            Tuple of (is_duplicate, original_placement_data)
        """
        state_hash = self.compute_state_hash(frame_path)
        is_dup, original = self.is_duplicate_state(state_hash, frame_number)

        if not is_dup and placement_data:
            self.register_placement(frame_path, frame_number, placement_data)

        return is_dup, original

    def get_placement_count(self) -> int:
        """Get number of unique placements registered."""
        return len(self.seen_states)

    def reset(self):
        """Reset tracker state."""
        self.seen_states.clear()
        self.placement_history.clear()


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
