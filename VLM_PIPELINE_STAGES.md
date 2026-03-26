# Video Enhancement VLM Pipeline - 5 Stages

This document describes the complete VLM call pipeline for processing LEGO assembly videos.

---

## Overview

The video enhancement pipeline uses **5 VLM call stages** to transform raw assembly videos into detailed step-by-step instructions:

1. **VLM Pass 1**: Frame Classification
2. **VLM Pass 2a**: Action Frame Analysis (sub-call within Pass 2)
3. **VLM Pass 2b**: Placement Validation
4. **VLM Pass 3**: Placement Reconciliation
5. **VLM Pass 4**: Atomic Sub-steps Generation

---

## Detailed Stage Descriptions

### VLM Pass 1: Frame Classification
**Location**: `_classify_frames_with_quality()`
**Prompt**: `prompts/video_frame_quality.txt`

**Purpose**: Classify each video frame into one of three categories:
- `placement_candidate`: Frame shows completed placement (hands clear, part placed)
- `action`: Frame shows hands actively moving/placing parts
- `irrelevant`: Frame not useful for instructions

**Input**: Batch of 8 frames at a time
**Output**: Classification + quality scores for each frame

**Example Output**:
```json
{
  "frame_type": "placement_candidate",
  "is_relevant": true,
  "quality_score": 0.95,
  "has_hand_obstruction": false,
  "confidence": 0.9
}
```

---

### VLM Pass 2a: Action Frame Analysis
**Location**: `_analyze_action_frames()`
**Prompt**: `prompts/video_action_frame_analysis.txt`

**Purpose**: Analyze action frames BETWEEN two placement candidates to determine:
- What LEGO part is being manipulated
- Whether it's a NEW placement (pickup) or just an ADJUSTMENT (touching already-placed part)

**Input**: All action frames between current and previous placement
**Output**: Part description + action type

**Example Output**:
```json
{
  "part_being_manipulated": "red round 2x2 tile",
  "color": "red",
  "part_type": "round tile",
  "size": "2x2",
  "action_type": "adjustment",  // or "pickup"
  "confidence": 0.95,
  "reasoning": "Part is already on the baseplate in first frame, just being pressed down"
}
```

**Key Logic**:
- **If NO action frames between placements** → Skip VLM call, mark as duplicate
- **If action_type == "adjustment"** → Warn Pass 2b that this is likely a duplicate
- **If action_type == "pickup"** → Tell Pass 2b this is likely a new placement

---

### VLM Pass 2b: Placement Validation
**Location**: `_validate_placements_with_context()`
**Prompt**: `prompts/video_placement_validation.txt` (enhanced with action frame context)

**Purpose**: Determine if a placement frame contains a NEW part or is a duplicate

**Input**:
- Previous placement frame image
- Current placement frame image
- Previous placement description
- **Action frame analysis from Pass 2a** (new!)
- Object count summary (from YOLO/SAM if enabled)

**Output**: Decision on whether to accept/reject the placement

**Example Output**:
```json
{
  "has_new_part": false,
  "is_duplicate_of_previous": true,
  "confidence": 1.0,
  "reasoning": "Action frames show red tile was just being adjusted (already placed)",
  "action_description": null
}
```

**Duplicate Detection Strategy**:
1. Check if action frames exist (if not → duplicate)
2. Check Pass 2a analysis (if adjustment → likely duplicate)
3. Compare images visually to confirm

---

### VLM Pass 3: Placement Reconciliation
**Location**: `_reconcile_single_placement()`
**Prompt**: `prompts/video_placement_reconciliation.txt`

**Purpose**: Verify detected parts against expected parts from enhanced.json

**Input**:
- Annotated frame with bounding box
- Cropped view of the part (tight crop for stud counting)
- Expected parts from manual for this step
- Reference images from manual

**Output**: Verified part information with corrections if needed

**Example Output**:
```json
{
  "verified": true,
  "matched_part": "white 1x2 brick",
  "video_detection_correct": true,
  "correction": null,
  "reasoning": "Part matches expected white 1x2 brick, stud count confirmed"
}
```

---

### VLM Pass 4: Atomic Sub-steps Generation
**Location**: `_generate_atomic_substeps()`
**Prompt**: `prompts/video_atomic_substeps.txt`

**Purpose**: Break down each manual step into atomic 1-part-per-sub-step instructions

**Input**:
- Manual step information
- Reconciled placements for this step
- Placement frame images

**Output**: Detailed sub-steps with spatial descriptions

**Example Output**:
```json
{
  "sub_step_number": 1,
  "action_type": "place",
  "parts_involved": ["white 1x2 brick"],
  "action_description": "Place white 1x2 brick on left side",
  "spatial_description": {
    "placement_part": "white 1x2 brick",
    "target_part": "grey baseplate",
    "location": "left side",
    "position_detail": "2 studs from left edge, 3 studs from bottom"
  }
}
```

---

## Pipeline Flow Diagram

```
Video Frames (every 5 frames)
    ↓
VLM PASS 1: Frame Classification
    ├─→ placement_candidate frames (60)
    └─→ action frames (124)
    ↓
For each placement candidate:
    ↓
    ├─→ Get action frames between current & previous placement
    ↓
    VLM PASS 2a: Action Frame Analysis
    │   ↓
    │   Determine: pickup vs adjustment
    ↓
    VLM PASS 2b: Placement Validation
    │   (uses Pass 2a results)
    │   ↓
    │   Accept or reject placement
    ↓
Accepted placements (15)
    ↓
For each placement:
    ↓
    VLM PASS 3: Reconciliation
    │   ↓
    │   Verify against manual parts
    ↓
Reconciled placements
    ↓
VLM PASS 4: Generate Atomic Sub-steps
    ↓
Final video_enhanced.json
```

---

## Key Improvements from Pass 2a Integration

**Problem Solved**: VLM was detecting already-placed parts as "new" when they were just being touched/adjusted

**Solution**: Pass 2a analyzes action frames to distinguish:
- **Pickup**: Part being lifted from table → New placement
- **Adjustment**: Part already on assembly → Not new, just being touched

**Impact**:
- ✅ Reduces false positives (duplicate detections)
- ✅ More accurate placement tracking
- ✅ Better understanding of assembly context
- ⚠️ Adds one extra VLM call per placement candidate (only when action frames exist)

---

## Logging

Detailed logs are written to:
```
data/processed/{manual_id}/vlm_reasoning_logs_{video_id}/placement_reasoning.log
```

Each placement candidate shows:
- Previous frame & placement context
- Action frames analyzed (count + frame numbers)
- Pass 2a results (part being manipulated, action type, reasoning)
- Pass 2b decision (accept/reject, reasoning)
- Final verdict

---

## Cost Considerations

- **Pass 1**: ~13-25 VLM calls (batches of 8 frames)
- **Pass 2a**: ~0-60 VLM calls (one per placement candidate with action frames)
- **Pass 2b**: ~60 VLM calls (one per placement candidate)
- **Pass 3**: ~15 VLM calls (one per accepted placement)
- **Pass 4**: ~1-8 VLM calls (one per manual step)

**Total**: ~90-170 VLM calls per video (depending on video length and duplicate rate)

Pass 2a adds cost but significantly improves accuracy by reducing false positives.
