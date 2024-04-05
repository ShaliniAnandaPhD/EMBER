def score_instruction(instruction, scenario, personas):
    safety_score = 0
    specificity_score = 0
    clarity_score = 0
    feedback_scores = []

    # Implement scoring logic for safety
    if "evacuate immediately" in instruction.lower():
        safety_score += 1
    if "follow designated evacuation routes" in instruction.lower():
        safety_score += 1
    if "stay calm" in instruction.lower():
        safety_score += 1

    # Implement scoring logic for specificity
    if scenario['location'] in instruction:
        specificity_score += 1
    if scenario['fire_name'] in instruction:
        specificity_score += 1
    if scenario['fire_year'] in instruction:
        specificity_score += 1

    # Implement scoring logic for clarity
    if "clear" in instruction.lower() or "concise" in instruction.lower():
        clarity_score += 1
    if "step-by-step" in instruction.lower():
        clarity_score += 1
    if "easy to understand" in instruction.lower():
        clarity_score += 1

    # Simulate user feedback based on personas
    for persona in personas:
        feedback_score = 0
        if persona['concern'] in instruction.lower():
            feedback_score += 1
        if persona['trait'] in instruction.lower():
            feedback_score += 1
        feedback_scores.append(feedback_score)

    # Normalize the scores
    safety_score = safety_score / 3
    specificity_score = specificity_score / 3
    clarity_score = clarity_score / 3
    feedback_score = sum(feedback_scores) / (len(personas) * 2)

    # Calculate the weighted total score
    weights = {'safety': 0.4, 'specificity': 0.3, 'clarity': 0.2, 'feedback': 0.1}
    total_score = (
        weights['safety'] * safety_score +
        weights['specificity'] * specificity_score +
        weights['clarity'] * clarity_score +
        weights['feedback'] * feedback_score
    )

    return total_score, safety_score, specificity_score, clarity_score, feedback_score

def score_instructions(instructions, scenario, personas):
    scored_instructions = []
    for instruction in instructions:
        total_score, safety_score, specificity_score, clarity_score, feedback_score = score_instruction(instruction, scenario, personas)
        scored_instructions.append({
            'instruction': instruction,
            'total_score': total_score,
            'safety_score': safety_score,
            'specificity_score': specificity_score,
            'clarity_score': clarity_score,
            'feedback_score': feedback_score
        })
    return scored_instructions