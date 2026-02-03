# templates for evaluation


TEMPLATE_MLLM_AS_JUDGE = """You are a professional judge for image generation, editing tasks. Given the original image, the edited image and the reference image in order, please evaluate the following generated image based on the criteria below. For each criterion, assign an integer score. Provide a short explanation for each score given. When evaluating, you can refer to the reference image, but do not use it as a strict guideline.

Original Question: {}

Correctness
Evaluate how well the generated image reflects the requested keypoints. Score according to how many key points are satisfied. For instance, if 2 key points are satisfied, the score is 2. If no key points are satisfied, the score is 0. If all key points are satisfied, the score is equal to the number of key points.

Keypoints: {}

Consistency with the Original Image
Evaluate if the generated image retains the key details and characteristics of the original image. This includes objects, colors, texts, and overall composition. Score between 0 and 5, where 0 means no consistency and 5 means perfect consistency.

Efficiency (Unnecessary Elements)
Evaluate if the generated image includes any elements or details that were not necessary or requested in the prompt. This can include extra objects, background elements, or any distracting features. Score between 0 and 5, where 0 means many unnecessary elements and 5 means no unnecessary elements.

Your answer should be in the following format:
Correctness: score 
Explanation of correctness: 

Consistency with the Original Image: score 
Explanation of consistency:

Efficiency: score 
Explanation of efficiency:
"""

TEMPLATE_CORRECTNESS_EDITING = """You are professional for judging image editing tasks.

## Requirements:
Evaluate how well the generated image reflects the requested keypoints mentioned in the input prompt.

## Input (Images in order):
- Original Question: {}
- Keypoints: {}
- Original Image
- Generated Image
- Reference Image 

## Details:
- The first image is the original image, the second is the generated one, and the third is a reference image showing one correct answer.
- Give **1 point per satisfied keypoint** based on the **second** image.
- If all keypoints are satisfied, the score equals the total number of keypoints.
- If none are satisfied, the score is 0.
- Reference Image (the third image, scores shouldn't be given to it!) can be used for understanding intent but **not as ground truth**.
- For diagrams, processes, if unnecessary steps (not the core steps to the task) are added, deduct **1 point for each.
- Partial satisfaction of a keypoint does **not** count as full credit unless clearly stated.
- Note: If the second image is almost the same as the original image and do not answer the question, or the second image does not give the correct format, the score is absolutely 0.
- If the second image try to change some details on the original image to satisfy a keypoint, this keypoint is regarded wrong.

## Output Format:
Explanation of correctness: [[Explain which keypoints were met or missed, with justification]]
Correctness: score"""

TEMPLATE_CORRECTNESS_GENERATION = """You are professional for judging image generation tasks.

## Requirements:
Evaluate how well the generated image reflects the requested keypoints mentioned in the input prompt.

## Input:
- Original Question: {}
- Keypoints: {}
- Generated Image

## Details:
- Give **1 point per satisfied keypoint**.
- If all keypoints are satisfied, the score equals the total number of keypoints.
- If none are satisfied, the score is 0.
- For diagrams, processes, if unnecessary steps are added, deduct **1 point.
- For web & UI design tasks, keypoints should be verified against the existence of matching textual indicators. If no textual information provided, no scores given.
- Reference Image can be used for understanding intent but **not as ground truth**.
- Partial satisfaction of a keypoint does **not** count as full credit unless clearly stated.

## Output Format:
Explanation of correctness: [[Explain which keypoints were met or missed, with justification]]
Correctness: score"""

TEMPLATE_VISUAL_EDIT = """You are professional for judging image editing tasks.

## Requirements:
Evaluate whether the generated image preserves important visual characteristics from the original image.

## Input:
- Original Image
- Generated Image

## Details:
Consider:
- Overall layout and structure
- Core objects and subjects
- Color scheme
- Presence of original texts or logos
- Style and lighting

Score from **0 to 5**:
- 5 = Perfectly consistent; nearly all visual aspects are preserved
- 4 = Largely consistent; small deviations that don't affect overall fidelity
- 3 = Moderately consistent; some key elements are preserved, others are altered
- 2 = Poor consistency; few elements preserved, major changes evident
- 1 = Almost no consistency; unrecognizable or mostly altered
- 0 = No consistency at all; entirely different image

## Output Format:
Explanation of consistency: [[Explain what was preserved and what was changed, with examples]]
Consistency: score"""

TEMPLATE_VISUAL_GENERATION = """You are professional for judging image generation tasks.
## Requirements:
Evaluate whether the generated image is visually and semantically coherent **within itself**. This includes logical structure, plausible anatomy, object relations, and internal consistency.

## Input:
- Generated Image

## Details:
Focus on the internal logic of the image:
- Are objects connected correctly (e.g., limbs, faces, items)?
- Are the texts in the generated image correct and meaningful?
- Is there a consistent perspective and lighting?
- Do body parts, backgrounds, and object placements make sense?
- Any obvious visual contradictions (e.g., two left hands, disconnected limbs, floating items)?

Score from **0 to 5**:
- 5 = Fully coherent; all visual elements are well-formed and logically arranged
- 4 = Mostly coherent; small flaws (e.g., slightly odd angle, but acceptable)
- 3 = Some inconsistencies; a few noticeable issues with structure or alignment
- 2 = Many inconsistencies; awkward or broken object relations
- 1 = Major incoherence; image has clear structural or logical errors
- 0 = Completely incoherent; image is broken, contradictory, or nonsensical

## Output Format:
Explanation of visual coherence: [[Point out any inconsistencies, broken structures, or logic issues]]
Visual Coherence: score"""

TEMPLATE_EFFICIENCY_GEN="""You are professional for judging image generation tasks.

## Requirements: Evaluate whether the generated image includes any unnecessary or unrequested elements that were not part of the original request.

## Input:
- Original Question: {} 
- Generated Image

## Details:
Look for:
- Extra objects, visual noise, or clutter
- Backgrounds or elements that were not requested
- Distracting additions that affect clarity or interpretation

Score from **0 to 5**:
- 5 = No unnecessary elements at all; clean and focused
- 4 = One very minor extra element, but not distracting
- 3 = Some small unnecessary elements, slightly distracting
- 2 = Noticeable extra content; affects image clarity
- 1 = Many extra elements; distracts heavily from the task
- 0 = Image is overwhelmed by unnecessary additions

## Output Format:
Explanation of efficiency: [[Explain if and what extra elements were introduced, and their impact]]
Efficiency: score"""

TEMPLATE_EFFICIENCY_EDIT="""You are professional for judging image editing tasks.

## Requirements: Evaluate whether the generated image includes any unnecessary or unrequested elements that were not part of the original request.

## Input:
- Original Question: {}
- Original Image 
- Generated Image

## Details:
Look for:
- Extra objects, visual noise, or clutter
- Backgrounds or elements that were not requested
- Distracting additions that affect clarity or interpretation

Score from **0 to 5**:
- 5 = No unnecessary elements at all; clean and focused
- 4 = One very minor extra element, but not distracting
- 3 = Some small unnecessary elements, slightly distracting
- 2 = Noticeable extra content; affects image clarity
- 1 = Many extra elements; distracts heavily from the task
- 0 = Image is overwhelmed by unnecessary additions

## Output Format:
Explanation of efficiency: [[Explain if and what extra elements were introduced, and their impact]]
Efficiency: score"""