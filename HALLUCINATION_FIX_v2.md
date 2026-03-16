# Hallucination Fix v2 - Frame 150 Issue

## Problem Report

**Frame 150** showed only a grey baseplate with a red SG50 tile, but the VLM hallucinated:

```json
{
  "action_description": "Add a white 1x2 brick to the left side of the grey baseplate",
  "new_parts": [
    {
      "part_description": "white 1x2 brick",
      "matched_from_manual": true,
      "color": "white",
      "type": "brick",
      "size": "1x2"
    }
  ],
  "confidence": 1.0
}
```

But the actual frame clearly shows **NO white brick** - only baseplate + red tile.

---

## Root Cause Analysis

### Issue 1: Perceptual Hash Check Ran AFTER VLM Call

**Before:**
```python
# Line 563-710 (OLD)
try:
    current_img = _resize_image(...)  # Load image
    current_b64 = _image_to_b64(current_img)

    # ... Build content with reference images ...

    messages = [{"role": "user", "content": content}]
    raw = self.vlm._litellm_with_retry(messages)  # ← EXPENSIVE VLM CALL
    result = _parse_json(raw)

    # THEN check for duplicates (TOO LATE!)
    is_hash_duplicate, original = self.state_tracker.check_and_register(...)
    if is_hash_duplicate:
        continue  # Already wasted the VLM call!
```

**Problem:**
- VLM was called even for frames that are perceptual duplicates
- When comparing two identical frames (frame 100 vs 150, both showing baseplate + red tile), VLM is forced to find a difference
- VLM hallucinates seeing a white brick because:
  - It's told to identify what **NEW** part was added
  - White bricks are in the `expected_parts_list`
  - VLM wants to be "helpful" and matches expected parts even when not visible

### Issue 2: Threshold Too Strict

```python
self.state_tracker = AssemblyStateTracker(hash_size=16, similarity_threshold=5)
```

Hamming distance of **5** is quite strict. Frames with minor lighting changes or slight camera angle shifts might not be caught as duplicates.

### Issue 3: Prompt Not Strong Enough

Old prompt said:
- ⚠️ **DO NOT HALLUCINATE**
- ⚠️ **BE CONSERVATIVE**

But didn't explicitly say: *"If the two frames look IDENTICAL, return has_new_part: false"*

---

## Fixes Applied

### Fix 1: Move Perceptual Hash Check BEFORE VLM Call

**New Code (lines 563-575):**
```python
try:
    # PRE-CHECK: Use perceptual hash to detect duplicates BEFORE expensive VLM call
    is_hash_duplicate, original = self.state_tracker.is_duplicate_state(
        self.state_tracker.compute_state_hash(current_frame["frame_path"]),
        frame_num
    )

    if is_hash_duplicate:
        logger.info(
            f"  Placement {i} (frame {frame_num}): "
            f"Duplicate detected via perceptual hash (matches frame {original['frame_number']}) - skipping VLM call"
        )
        continue  # Skip the VLM call entirely!

    current_img = _resize_image(str(current_frame["frame_path"]), width=800)
    current_b64 = _image_to_b64(current_img)
    # ... continue with VLM call only if NOT a duplicate ...
```

**Benefits:**
- Saves VLM calls on duplicate frames (reduces cost & latency)
- VLM never sees identical/near-identical frame pairs
- Prevents forced hallucination scenarios

**Also removed redundant post-VLM hash check (lines 700-702)** since we're now doing it before.

### Fix 2: Relaxed Similarity Threshold

**Changed (line 100):**
```python
# Before:
self.state_tracker = AssemblyStateTracker(hash_size=16, similarity_threshold=5)

# After:
self.state_tracker = AssemblyStateTracker(hash_size=16, similarity_threshold=10)
```

**Rationale:**
- Hamming distance of **10** is more forgiving for:
  - Minor lighting variations
  - Slight camera angle changes
  - Small hand movements in/out of frame
- Still strict enough to differentiate when actual parts are added

### Fix 3: Strengthened Prompt

**Added to `prompts/video_placement_validation.txt` (lines 10-12):**
```
⚠️ **IDENTICAL FRAMES = NO NEW PART** - If the previous and current frames show the EXACT SAME assembly state (even if minor lighting/angle differences), you MUST set has_new_part: false
⚠️ **NO GUESSING** - Do not infer or predict parts based on the expected parts list. Only report what you ACTUALLY SEE in the current frame.
```

**Emphasis:** VLM should not "predict" or "infer" parts from the expected parts list.

---

## Expected Results

### Before (Frame 150 Issue):

```
Pass 1: Frame 150 classified as "placement_candidate" (quality: 0.95, no hands)
Pass 2:
  - Compare frame 100 (baseplate + tile) vs frame 150 (baseplate + tile)
  - VLM sees they're similar but is told to find NEW part
  - VLM hallucinates: "white 1x2 brick added" ← WRONG!
  - Perceptual hash checks AFTER (too late)
  - Placement accepted with confidence 1.0
```

### After (With Fixes):

```
Pass 1: Frame 150 classified as "placement_candidate" (quality: 0.95, no hands)
Pass 2:
  - Compute perceptual hash of frame 150
  - Compare with frame 100's hash
  - Hamming distance = 3 (well below threshold of 10)
  - LOG: "Duplicate detected via perceptual hash (matches frame 100) - skipping VLM call"
  - Frame 150 rejected WITHOUT wasting VLM call
  - No hallucination possible!
```

### Cost Savings

Assuming ~40 placement candidates and ~30% are duplicates:
- **Before:** 40 VLM calls (some hallucinate on duplicates)
- **After:** ~28 VLM calls (duplicates filtered pre-VLM)
- **Savings:** ~30% fewer VLM calls + higher accuracy

---

## Testing Instructions

Cache cleared for fresh run:
```bash
rm data/processed/111111/video_frame_quality_changi_airport.json
rm data/processed/111111/video_validated_placements_changi_airport.json
rm data/processed/111111/video_enhanced_v2_changi_airport.json
```

Run the test:
```bash
uv run python test_video_enhancer_v2.py
```

### What to Verify:

1. **Frame 150 should be rejected:**
   ```
   Placement 14 (frame 150): Duplicate detected via perceptual hash
   (matches frame 100) - skipping VLM call
   ```

2. **No hallucinated white bricks** in early placements

3. **Logs should show:**
   ```
   Pass 2 cache: Validating X placement candidates...
   Placement 0 (frame 40): Place grey baseplate
   Placement 4 (frame 100): Add red tile
   Placement 14 (frame 150): Duplicate detected - skipping VLM call  ← NEW!
   Placement 19 (frame 210): Add first white brick  ← ACTUAL white brick
   ```

4. **Step tracking should be accurate:**
   - Step 1 should complete after 9 individual parts (not 4)
   - Check logs for: "Step 1 complete (9/9 parts placed)"

5. **Run visualization:**
   ```bash
   uv run python visualize_video_pipeline.py
   ```
   - Frame 150 should NOT appear in Pass 2 validated placements
   - Only frames with actual new parts should be shown

---

## Files Modified

1. **`backend/services/video_enhancer_v2.py`**
   - Lines 563-575: Added pre-VLM perceptual hash check
   - Line 100: Increased similarity threshold from 5 to 10
   - Lines 700-702: Removed redundant post-VLM hash check

2. **`prompts/video_placement_validation.txt`**
   - Lines 10-12: Added stronger warnings about identical frames and no guessing

---

## Additional Notes

### Why This Happens with VLMs

Vision Language Models (VLMs) are trained to be helpful and find patterns. When given:
- Two very similar images
- An instruction to "find what NEW part was added"
- A list of expected parts that SHOULD be added soon

The VLM may hallucinate seeing the expected part even if it's not there, especially when:
- The images are nearly identical (forcing the model to "look harder")
- The expected part is highly likely based on context
- The model has high confidence in the manual's accuracy

**Prevention:** Don't ask the VLM to compare identical frames! Filter duplicates BEFORE the VLM call.

### Perceptual Hashing vs VLM Comparison

| Method | Speed | Cost | Accuracy for Duplicates | When to Use |
|--------|-------|------|------------------------|-------------|
| Perceptual Hash | Very fast | Free | Excellent (>95%) | Pre-filter duplicates |
| VLM Comparison | Slow (2-3s) | $$$ | Good but prone to hallucination | Only for non-duplicates |

**Best Practice:** Use perceptual hash as a **gatekeeper** before expensive VLM calls.
