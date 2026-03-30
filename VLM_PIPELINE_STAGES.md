# Video Enhancement VLM Pipeline - Three-Pass Architecture with SAM3

This document describes the complete VLM call pipeline for processing LEGO assembly videos with the three-pass validation architecture and inline reconciliation.

---

## Overview

The video enhancement pipeline uses **6 VLM call stages** to transform raw assembly videos into detailed step-by-step instructions:

1. **VLM Pass 1**: Frame Classification
2. **VLM Pass 2a**: Action Sequence Analysis (prev → ALL action frames → current, NO SAM3)
3. **VLM Pass 2b**: SAM3 Visual Comparison (ONLY SAM3 segmented prev and current)
4. **VLM Pass 2c**: Multi-Source Reconciliation (Pass 2a + Pass 2b + Expected Parts)
5. **VLM Pass 2d**: Step Completion Verification (Compare assembly state vs expected subassembly)
6. **VLM Pass 3**: Atomic Sub-steps Generation

**Key improvements:**
- ✅ **Four Validation Sub-Passes**: Pass 2a (action analysis), Pass 2b (SAM3 comparison), Pass 2c (reconciliation), Pass 2d (step completion)
- ✅ **Temporal Context**: Pass 1 uses temporal sequences to detect when hands retreat from placement
- ✅ **All Action Frames**: Pass 2a includes ALL action frames (no sampling limit)
- ✅ **SAM3 Segmentation**: Pass 2b uses clean SAM3-segmented images for accurate stud counting
- ✅ **Multi-Source Reconciliation**: Pass 2c reconciles THREE sources of evidence for final answer
- ✅ **Immediate Reconciliation**: Pass 2c runs AFTER EACH placement (not at the end), fixing errors before they propagate
- ✅ **Step Advancement**: Pass 2d verifies assembly completion and determines when to advance to next step

---

## Detailed Stage Descriptions

### VLM Pass 1: Frame Classification
**Location**: `_classify_frames_with_quality()`
**Prompt**: `prompts/video_frame_quality.txt`

**Purpose**: Classify each video frame into one of three categories:
- `placement_candidate`: Frame shows completed placement (hands clear, part placed)
- `action`: Frame shows hands actively moving/placing parts
- `irrelevant`: Frame not useful for instructions

**Input**: Batch of 8 frames at a time (in chronological order)
**Output**: Classification + quality scores for each frame

**Key Features**:
- **Temporal Context**: Frames are analyzed in chronological sequence, allowing the VLM to use temporal patterns
- **"Hands Retreated" Criteria**: PRIMARY requirement for placement candidates is that "hands have largely RETREATED out of view"
- **Clear Assembly View**: Must be able to see the main assembly clearly (not obscured by hands)
- **Strict Disqualification**: If hands are still prominently in frame or assembly is obscured → classified as ACTION, not placement

**Example Output**:
```json
{
  "frame_type": "placement_candidate",
  "is_relevant": true,
  "quality_score": 0.95,
  "has_hand_obstruction": false,
  "is_stable": true,
  "confidence": 0.95,
  "reasoning": "Hands have retreated, clear view of main assembly with newly placed part visible"
}
```

**Improvements from Previous Version**:
- Previously accepted frames where hands were still visible pressing down on parts
- Now requires hands to have retreated before classifying as placement_candidate
- Uses temporal sequence to understand: action (hands placing) → placement (hands retreated) → stable (assembly visible)

---

### VLM Pass 2a: Action Sequence Analysis
**Location**: `_pass2a_action_analysis()`
**Prompt**: `prompts/video_pass2a_action_analysis.txt`

**Purpose**: Analyze the action sequence to understand what is happening:
- What action is being performed?
- Is a new part being added, or is an existing part being adjusted?
- What part appears to be involved (from action frames)?
- What is the action narrative?

**Input (Multimodal Sequence)**:
1. **Previous Placement Frame** (original, NO SAM3)
   - Shows last accepted placement state
2. **ALL Action Frames** (NO sampling limit)
   - Shows hands manipulating parts between placements
   - Previously limited to 5 frames, now includes ALL action frames
3. **Current Placement Frame** (original, NO SAM3)
   - Shows current candidate placement state

**Output**:
```json
{
  "action_type": "place_new_part",
  "has_new_part": true,
  "part_from_actions": {
    "description": "dark grey brick, appears to be 2x2",
    "color": "dark grey",
    "type": "brick",
    "approximate_size": "2x2",
    "confidence": 0.8
  },
  "box_2d": [350, 600, 420, 700],
  "action_narrative": "Hands pick up a dark grey brick from the side, position it over the baseplate, and place it down between two white bricks.",
  "confidence": 0.9,
  "reasoning": "Action frames clearly show hands holding a dark grey brick piece and placing it onto the assembly. Bounding box drawn around the new part in the final placement frame."
}
```

**Key Features**:
- Focuses on understanding the ACTION from the sequence
- Describes what is seen in the hands during action frames
- Does NOT use SAM3 (analyzes original video frames)
- Provides approximate part identification (Pass 2b and 2c will verify)
- **Generates bounding box (box_2d)** for the new part in the CURRENT PLACEMENT FRAME
  - Coordinates are for the original frame (not SAM3)
  - Used for annotating original frames and cropping for Pass 2c

---

### VLM Pass 2b: SAM3 Visual Comparison
**Location**: `_pass2b_sam3_comparison()`
**Prompt**: `prompts/video_pass2b_sam3_comparison.txt`

**Purpose**: Compare SAM3-segmented images to identify what changed:
- Is there a new part added, or is this the same assembly state?
- What is the new part? (color, type, size, stud count)
- Where is the new part located?
- Bounding box around the new part

**Input (Multimodal - 2 Images Only)**:
1. **Previous Placement (SAM3 segmented)**
   - Assembly cropped on white background
2. **Current Placement (SAM3 segmented)**
   - Assembly cropped on white background

**NO action frames** - This pass only compares the before/after assembly states

**SAM3 Segmentation**:
- Runs Roboflow SAM3 API with prompt: `"lego assembly"`
- Detects the entire LEGO assembly region as one object
- Crops out the assembly and places it on a white background
- Removes noise: table, hands, and other background objects
- Saved to: `data/processed/{manual_id}/sam3_segmented_{video_id}/`
- Falls back to original images if SAM3 fails or API key not set
- **Why white background?** Eliminates distractions and focuses VLM on stud counting

**Output**:
```json
{
  "is_duplicate": false,
  "has_new_part": true,
  "what_changed": "A new dark grey 2x2 brick was added to the left side of the baseplate",
  "new_part_detected": {
    "description": "dark grey 2x2 brick",
    "color": "dark grey",
    "type": "brick",
    "size": "2x2",
    "stud_count": 4
  },
  "spatial_position": {
    "location": "left side",
    "reference_object": "grey baseplate",
    "orientation": "upright"
  },
  "box_2d": [400, 100, 650, 300],
  "confidence": 0.95,
  "reasoning": "PREVIOUS image shows baseplate with 2 white bricks. CURRENT image shows baseplate with 2 white bricks PLUS one additional dark grey brick on the left side. STUD COUNT: 4 studs in 2x2 grid."
}
```

**Key Features**:
- **CRITICAL**: Counts studs to determine size (2 studs = 1x2, 4 studs = 2x2)
- Clean SAM3 images improve stud counting accuracy
- Focuses on visual comparison, not action understanding
- Determines duplicate status (if no change detected)
- **Generates bounding box (box_2d)** for the new part in the CURRENT SAM3 FRAME
  - Coordinates are for the SAM3-segmented frame (not original)
  - Used for annotating SAM3 frames and cropping for Pass 2c

---

### VLM Pass 2c: Multi-Source Reconciliation
**Location**: `_pass2c_reconcile_all_sources()`
**Prompt**: `prompts/video_pass2c_reconciliation.txt`

**Purpose**: Reconcile THREE sources of evidence to determine the TRUE identity of the placed part:
1. **Pass 2a Result**: What the VLM saw in the action sequence
2. **Pass 2b Result**: What the VLM saw by comparing SAM3 images
3. **Expected Parts List**: What the manual says should be used in this step

Runs IMMEDIATELY after each accepted placement, correcting errors before they propagate.

**Input**:
- Pass 2a result (action analysis)
- Pass 2b result (SAM3 comparison)
- Expected parts from manual for this step
- Visual evidence:
  - VIEW 1: Cropped from ORIGINAL frame using Pass 2a's box_2d (shows placement context on assembly)
  - VIEW 2: Cropped from SAM3 frame using Pass 2b's box_2d (shows isolated part on white background)
  - VIEW 3: Reference images of expected parts from manual

**Note on Bounding Boxes**:
- Pass 2a's `box_2d_original`: Coordinates for cropping the original frame (includes hands, table, context)
- Pass 2b's `box_2d_sam3`: Coordinates for cropping the SAM3-segmented frame (clean white background)
- Pass 2c uses the appropriate box for each crop to ensure correct coordinate systems

**Output**:
```json
{
  "stud_count_analysis": {
    "visible_studs": 4,
    "arrangement": "2x2 grid (square pattern)",
    "counting_confidence": 0.95
  },
  "sources_agree": false,
  "pass2a_correct": true,
  "pass2b_correct": false,
  "expected_parts_match": true,
  "final_part": {
    "description": "dark grey 2x2 brick",
    "color": "dark grey",
    "type": "brick",
    "size": "2x2",
    "stud_count": 4,
    "confidence": 0.95
  },
  "corrections": {
    "pass2a_said": "dark grey brick, appears to be 2x2",
    "pass2b_said": "dark grey 1x2 brick",
    "expected_said": "dark grey 2x2 brick",
    "actual_part": "dark grey 2x2 brick",
    "pass2a_error": null,
    "pass2b_error": "Pass 2b incorrectly counted studs - said 1x2 but VIEW 2 shows 4 studs in 2x2 grid",
    "expected_parts_error": null
  },
  "reasoning": "STUD COUNT: VIEW 2 clearly shows 4 studs arranged in a 2x2 square grid. Pass 2a said 'appears 2x2' (CORRECT), Pass 2b said '1x2' (WRONG), expected parts include '2x2 brick' (matches). Pass 2b made a stud counting error."
}
```

**Key Features**:
- **Independent Verification**: Counts studs in VIEW 2 (cropped close-up) directly
- **No Bias**: Doesn't favor Pass 2a, 2b, or expected parts - verifies all sources
- **Explicit Conflicts**: Clearly states which source was wrong and why
- **Final Authority**: The `final_part` field is the authoritative answer after reconciliation

**Impact**:
- ✅ Errors fixed immediately (e.g., grey 1×2 → grey 2×2)
- ✅ Corrected description replaces original in action_data
- ✅ Next placement sees CORRECTED context, not wrong context
- ✅ Catches errors from BOTH Pass 2a and Pass 2b

---

### VLM Pass 2d: Step Completion Verification
**Location**: `_pass2d_verify_step_completion()`
**Prompt**: `prompts/video_pass2d_step_completion.txt`

**Purpose**: Verify whether the current LEGO assembly has reached the completion state for a given step by comparing the current assembly against the expected subassembly image(s) for that step.

**Input (Multimodal)**:
1. **Current Assembly (SAM3-segmented)** - The current state of the assembly on white background
2. **Expected Subassembly Image(s)** - Reference image(s) showing what the assembly should look like at step completion
3. **Step Number** - The current step being verified
4. **Expected Parts List** - List of parts that should be present in this step

**Output**:
```json
{
  "is_step_complete": true,
  "current_step_verified": 1,
  "should_advance_to_step": 2,
  "parts_matched": ["dark grey 2x2 brick", "dark grey 2x2 brick", "dark grey 2x2 brick"],
  "confidence": 0.95,
  "reasoning": "The current assembly matches the expected Step 1 completion state. All 3 dark grey 2x2 bricks are present and correctly positioned horizontally on the baseplate. The structural arrangement matches the reference image despite a slightly different camera angle.",
  "discrepancies": null
}
```

**If step is NOT complete**:
```json
{
  "is_step_complete": false,
  "current_step_verified": 1,
  "should_advance_to_step": 1,
  "parts_matched": ["dark grey 2x2 brick", "dark grey 2x2 brick"],
  "confidence": 0.90,
  "reasoning": "The current assembly shows only 2 dark grey bricks placed. The expected Step 1 completion requires 3 bricks. Missing the third brick on the right side.",
  "discrepancies": {
    "missing_parts": ["dark grey 2x2 brick (third brick)"],
    "incorrect_positions": null,
    "extra_parts": null
  }
}
```

**Key Features**:
- **Structural Matching**: Focuses on structural match, not pixel-perfect comparison
- **Step Advancement Logic**: Determines when to advance to the next step based on assembly state
- **Handles Camera Angles**: Accepts minor camera angle and lighting variations
- **Part Verification**: Confirms all expected parts for the step are present
- **Runs After Each Placement**: Checks after each accepted placement to determine if step is complete

**Impact**:
- ✅ Accurate step advancement (e.g., frame 530 correctly advances to step 2 after step 1 completion)
- ✅ Prevents premature advancement (stays on current step until all required parts are placed)
- ✅ Robust to detection errors (uses visual assembly state, not just part counting)
- ✅ Handles assembly progression (detects when assembly has progressed beyond current step)

---

### VLM Pass 3: Atomic Sub-steps Generation
**Location**: `_generate_atomic_substeps()`
**Prompt**: `prompts/video_atomic_substeps.txt`

**Purpose**: Break down each manual step into atomic 1-part-per-sub-step instructions

**Input**:
- Manual step information
- **Validated** placements for this step (already corrected by Pass 2c reconciliation)
- Placement frame images

**Output**:
```json
{
  "sub_step_number": 1,
  "action_type": "place",
  "parts_involved": ["dark grey 2x2 brick"],  // Uses reconciled description from Pass 2c
  "action_description": "Place dark grey 2x2 brick on left side",
  "spatial_description": {
    "placement_part": "dark grey 2x2 brick",
    "target_part": "grey baseplate",
    "location": "left side",
    "position_detail": "2 studs from left edge, 3 studs from bottom"
  },
  "verified": true  // From Pass 2c reconciliation
}
```

**Key Features**:
- Uses validated placements that already contain Pass 2c reconciliation data
- No longer needs a separate reconciliation pass (old "Pass 3" was removed as redundant)
- Generates atomic sub-steps with spatial positioning details

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
    ├─→ Get previous accepted placement frame
    ├─→ Get ALL action frames between previous and current (no sampling)
    ↓
    VLM PASS 2a: Action Sequence Analysis
    │   Input: prev → ALL action frames → current (NO SAM3)
    │   Output: action_type, part_from_actions, action_narrative, box_2d_original
    ↓
    VLM PASS 2b: SAM3 Visual Comparison
    │   Input: ONLY SAM3 segmented prev and current (NO action frames)
    │   Output: is_duplicate, new_part_detected, box_2d_sam3
    ↓
    Check duplicate status from Pass 2b
    ↓
    IF NOT DUPLICATE:
        ↓
        VLM PASS 2c: Multi-Source Reconciliation
        │   Input: Pass 2a + Pass 2b + Expected Parts + Visual Evidence
        │   Output: final_part (reconciled answer)
        │   ↓
        │   Accept placement with reconciled data
        │   Update action_description with final_part
        ↓
        VLM PASS 2d: Step Completion Verification
        │   Input: Current SAM3 assembly + Expected subassembly + Step info
        │   Output: is_step_complete, should_advance_to_step
        │   ↓
        │   IF step complete → Advance to next step
        ↓
        Continue to next placement candidate
    ↓
All placements validated + reconciled + step advancement handled
    ↓
VLM PASS 3: Generate Atomic Sub-steps
    (Uses validated placements with Pass 2c reconciliation)
    ↓
Final video_enhanced.json
```

---

## Key Improvements from Three-Pass Architecture

### Improvement 1: Separate Action Analysis from Visual Comparison
**Why Four Sub-Passes in Pass 2**:
- **Pass 2a**: Focuses on understanding the ACTION (what's happening in the sequence)
  - Sees ALL action frames (no sampling limit)
  - Provides action narrative and approximate part identification
  - Generates bounding boxes for original frames
- **Pass 2b**: Focuses on VISUAL COMPARISON (what changed)
  - Uses clean SAM3 images for accurate stud counting
  - Determines duplicate status and exact part identification
  - Generates bounding boxes for SAM3 frames
- **Pass 2c**: RECONCILES all sources to find the truth
  - Counts studs independently in cropped close-up (VIEW 2)
  - Explicitly identifies which source was wrong and why
  - Uses correct bounding boxes for each coordinate system
- **Pass 2d**: VERIFIES step completion
  - Compares assembly state against expected subassembly
  - Determines when to advance to next step

**Benefit**:
- Each sub-pass has a clear, focused purpose
- Pass 2c can catch errors from BOTH Pass 2a and Pass 2b
- Pass 2d ensures accurate step advancement
- No "split-brain" problem - reconciliation sees both perspectives
- Dual bounding boxes prevent coordinate system mismatches

### Improvement 2: Multi-Source Reconciliation
**Three Sources of Evidence**:
1. **Pass 2a**: What the action sequence shows
2. **Pass 2b**: What the SAM3 comparison shows
3. **Expected Parts**: What the manual says should be used

**How It Works**:
- Pass 2c independently counts studs in VIEW 2 (cropped close-up)
- Compares all three sources: Do they agree?
- If sources disagree → explicitly states who was wrong
- Returns final authoritative answer in `final_part`

**Example**:
```
Pass 2a: "dark grey brick, appears to be 2x2"
Pass 2b: "dark grey 1x2 brick"  ← WRONG
Expected: "dark grey 2x2 brick"
Pass 2c: Counts 4 studs in VIEW 2 → "dark grey 2x2 brick" (CORRECT)
         Explicitly states: "Pass 2b made a stud counting error"
```

### Improvement 3: Immediate Error Correction
**Old Behavior**:
- Frame 400: VLM detects "dark grey 1×2" (wrong)
- Frame 420: Prompt includes "Previous: dark grey 1×2"
- VLM uses wrong context → makes another error

**New Behavior**:
- Frame 400: Pass 2b detects "dark grey 1×2" (wrong)
- **Pass 2c immediately fixes it:** "dark grey 2×2" (correct)
- Frame 420: Prompt includes "Previous: dark grey 2×2" (corrected)
- VLM has CORRECT context → makes better decision

### Improvement 4: All Action Frames Included
**Old Behavior**:
- Action frames were sampled (limited to 5 frames)
- VLM might miss critical moments in the action

**New Behavior**:
- Pass 2a includes ALL action frames (no sampling)
- VLM sees complete action sequence
- Better understanding of what's happening

### Improvement 5: SAM3 for Accurate Stud Counting
**How It Works**:
- Pass 2b uses SAM3 to segment the assembly and remove background noise
- Assembly is cropped and placed on white background
- VLM can focus on counting studs without distractions

**Benefit**:
- Improves accuracy for 1×2 vs 2×2 vs 2×4 detection
- Clean images help VLM count studs more reliably

### Improvement 6: Visual Step Completion Verification
**Old Behavior**:
- Step advancement based on part counting
- Prone to errors if parts misdetected or duplicates not filtered
- Frame 530 should advance to step 2 but doesn't

**New Behavior (Pass 2d)**:
- Compares current SAM3 assembly against expected subassembly image
- Uses visual structural matching instead of just counting parts
- Explicitly determines: is_step_complete and should_advance_to_step
- Frame 530 correctly advances to step 2 when Step 1 assembly matches expected state

**Benefit**:
- More robust to part detection errors
- Uses visual assembly state as ground truth
- Handles camera angle variations
- Prevents premature or late step advancement

### Improvement 7: Dual Bounding Box System
**The Problem**:
- Pass 2a analyzes original frames (with hands, table, background)
- Pass 2b analyzes SAM3 frames (assembly only, white background)
- Different coordinate systems → can't use same bounding box!

**The Solution**:
- Pass 2a generates `box_2d_original` for original frame coordinate system
- Pass 2b generates `box_2d_sam3` for SAM3 frame coordinate system
- Pass 2c uses correct box for each crop operation
- Annotated frames use correct box for their frame type

**Benefit**:
- Bounding boxes now appear correctly on LEGO parts (not hands/shadows)
- VIEW 1 and VIEW 2 in Pass 2c are cropped correctly
- No more coordinate system mismatch errors

### Improvement 8: Temporal Context in Frame Classification
**Old Behavior**:
- Pass 1 classified frames independently
- Accepted frames where hands were still visible pressing down
- Frames 0605, 0380 incorrectly classified as placement_candidate

**New Behavior**:
- Pass 1 receives frames in chronological sequence
- Uses temporal context: action (hands placing) → placement (hands retreated) → stable
- PRIMARY CRITERIA: "Hands have largely RETREATED out of view"
- Frames with hands still prominent → classified as ACTION, not placement

**Benefit**:
- Better placement candidate detection
- Waits for hands to retreat before capturing placement frame
- More stable, clear images for placement analysis

---

## Cost Considerations

- **Pass 1**: ~13-25 VLM calls (batches of 8 frames)
- **Pass 2a**: ~0-60 VLM calls (one per placement candidate with action frames)
  - Includes ALL action frames (no sampling limit)
  - Does NOT use SAM3 (original frames only)
  - Generates bounding boxes for original frames
- **Pass 2b**: ~0-60 VLM calls (one per placement candidate)
  - ONLY sends 2 SAM3 images (prev + current)
  - Requires SAM3 API calls for segmentation (~0-120 calls)
  - Generates bounding boxes for SAM3 frames
- **Pass 2c**: ~15 VLM calls (one per accepted placement, inline)
  - Reconciles Pass 2a + Pass 2b + expected parts
  - Runs immediately after each accepted placement
  - Uses both bounding boxes for cropping
- **Pass 2d**: ~15 VLM calls (one per accepted placement, inline)
  - Verifies step completion by comparing assembly state vs expected subassembly
  - Determines when to advance to next step
- **Pass 3**: ~1-8 VLM calls (one per manual step)

**Total VLM**: ~120-183 VLM calls per video
**Total SAM3**: ~0-120 API calls (for Pass 2b segmentation)

**Cost Breakdown**:
- Pass 1: 13-25 VLM calls (frame classification)
- Pass 2a: 0-60 VLM calls (action analysis with bounding boxes)
- Pass 2b: 0-60 VLM calls (SAM3 comparison with bounding boxes)
- Pass 2c: 15 VLM calls (reconciliation)
- Pass 2d: 15 VLM calls (step completion verification)
- Pass 3: 1-8 VLM calls (atomic sub-steps)
- SAM3: 0-120 API calls

**Trade-off**: More VLM calls than unified architecture, but better accuracy through:
- Temporal context in Pass 1 (hands retreated detection)
- Complete action sequences (all frames)
- Clean SAM3 images for stud counting
- Separate bounding boxes for original vs SAM3 frames
- Multi-source reconciliation catching errors from both passes
- Visual step completion verification

---

## Logging and Debugging

### Detailed Logs
**Location**: `data/processed/{manual_id}/vlm_reasoning_logs_{video_id}/placement_reasoning.log`

Each placement candidate shows:
- Previous frame & placement context
- Action frames analyzed (count + frame numbers)
- Current frame info
- **VLM Pass 2a - Action Analysis**:
  - Analyzed N action frames
  - action_type, has_new_part
  - part_from_actions (description, color, type, size)
  - box_2d (for original frame)
  - action_narrative
  - confidence, reasoning
- **VLM Pass 2b - SAM3 Comparison**:
  - SAM3 segmented image paths (prev + current)
  - is_duplicate, has_new_part
  - new_part_detected (description, stud_count)
  - spatial_position, box_2d (for SAM3 frame)
  - confidence, reasoning
- **VLM Pass 2c - Reconciliation** (if accepted):
  - Inputs: Pass 2a result, Pass 2b result, Expected parts
  - sources_agree, pass2a_correct, pass2b_correct, expected_parts_match
  - final_part (reconciled answer)
  - corrections (what each source said, which were wrong)
  - stud_count_analysis (independent verification)
  - reasoning
- **VLM Pass 2d - Step Completion** (if accepted):
  - Inputs: Current SAM3 assembly, Expected subassembly, Step info
  - is_step_complete, should_advance_to_step
  - parts_matched, discrepancies
  - confidence, reasoning
- Final verdict

### SAM3 Segmented Frames
**Location**: `data/processed/{manual_id}/sam3_segmented_{video_id}/`
- Shows the LEGO assembly cropped out and placed on white background
- Removes all noise: table, hands, background objects
- Preserves original assembly colors for accurate VLM analysis
- Helps visualize the clean, focused view the VLM receives
- Useful for debugging detection and stud counting

### Annotated Placement Frames
**Location**: `data/processed/{manual_id}/validated_placement_annotated_{video_id}/`
- Shows bounding boxes around detected parts
- Color-coded by confidence (green=high, orange=medium, red=low)

---

## Running the Pipeline

```bash
# Set environment variables
export GEMINI_API_KEY="your-gemini-key"
export ROBOFLOW_API_KEY="your-roboflow-key"  # For SAM3

# Run test
uv run python test_video_enhancer_v2.py
```

**Output**:
1. SAM3 segmented frames (if ROBOFLOW_API_KEY set)
2. Annotated placement frames
3. VLM reasoning logs
4. Final `video_enhanced_v2_{video_id}.json`

---

## Configuration

### Settings
**File**: `config/settings.py`

```python
# VLM Configuration
gemini_api_key: Optional[str] = None
vlm_model: str = "gemini/gemini-robotics-er-1.5-preview"
placement_min_confidence: float = 0.6

# Roboflow Configuration (for SAM3)
roboflow_api_key: Optional[str] = None
```

### Disabling SAM3
SAM3 segmentation is optional:
- If `roboflow_api_key` is NOT set → Pass 2b returns None and validation is skipped
- If SAM3 API call fails → Pass 2b falls back to original frames
- Pass 2a always uses original frames (does not use SAM3)

---

## Troubleshooting

### SAM3 Segmentation Not Working
1. Check `ROBOFLOW_API_KEY` is set in `.env`
2. Check SAM3 segmented frames directory exists and has images
3. Check logs for SAM3 API errors
4. If Pass 2b is failing, validation will be skipped (requires SAM3)

### VLM Detections Still Wrong
1. Check all four validation sub-passes are running:
   - VLM PASS 2a - ACTION ANALYSIS
   - VLM PASS 2b - SAM3 COMPARISON
   - VLM PASS 2c - RECONCILIATION
   - VLM PASS 2d - STEP COMPLETION VERIFICATION
2. Check Pass 2c reconciliation output:
   - sources_agree: Are the three sources consistent?
   - pass2a_correct, pass2b_correct: Which source was wrong?
   - corrections: What corrections were made?
3. Check Pass 2d step completion output:
   - is_step_complete: Is the step verified as complete?
   - should_advance_to_step: Is advancement happening correctly?
   - discrepancies: Are any parts missing or incorrectly positioned?
4. Review VLM reasoning logs for detailed decision process

### Error Propagation Still Happening
1. Verify Pass 2c runs AFTER each placement (not batched at end)
2. Check that `final_action_desc` uses Pass 2c's `final_part` result
3. Check that next placement's "PREVIOUS PLACEMENT CONTEXT" shows corrected description
4. Verify action_data is updated with reconciled information

### Pass 2a and Pass 2b Disagree
This is expected! Pass 2c is designed to handle disagreements:
1. Check Pass 2c logs for reconciliation reasoning
2. Pass 2c independently counts studs in VIEW 2 (cropped close-up)
3. Pass 2c explicitly states which source was wrong in `corrections` field
4. Use `final_part` from Pass 2c as the authoritative answer

---

## Summary

The three-pass architecture with SAM3 and immediate reconciliation provides:

✅ **Temporal Context**: Pass 1 uses frame sequences to detect when hands have retreated
✅ **Separate, Focused Analysis**: Pass 2a (action), Pass 2b (visual), Pass 2c (reconciliation), Pass 2d (step completion)
✅ **Dual Bounding Boxes**: Pass 2a generates box for original frames, Pass 2b for SAM3 frames (correct coordinate systems)
✅ **Multi-Source Reconciliation**: Pass 2c reconciles THREE sources of evidence for final answer
✅ **All Action Frames**: Pass 2a includes ALL action frames (no sampling limit)
✅ **Better Stud Counting**: SAM3 segmentation in Pass 2b improves accuracy
✅ **Error Detection**: Pass 2c can catch errors from BOTH Pass 2a and Pass 2b
✅ **Immediate Error Correction**: Pass 2c runs after each placement, fixing errors before propagation
✅ **Explicit Conflict Resolution**: Pass 2c states which source was wrong and why
✅ **Visual Step Completion**: Pass 2d verifies assembly state matches expected subassembly for accurate step advancement
✅ **No Redundant Passes**: Old "Pass 3" (per-placement reconciliation) removed - Pass 2c already handles this inline
✅ **Better Debugging**: Detailed logs show all pass results + corrections + step verification
