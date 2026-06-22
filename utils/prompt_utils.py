def set_prompts(batch_size, data='exp2'):
    if data == 'exp3':
        # Generate all 6 prompt combinations
        shade_list = ['dark', 'light']
        color_list = ['blue', 'purple', 'yellow']
        prompt_list = [f"A {shade} {color} dry film residual defect." for shade in shade_list for color in color_list]

    elif data == 'exp2':
        prompt_list = ['A blue dry film residual defect.']

    elif data == 'exp4':
        prompt_list = ['A large scratch defect covering a wide area']
        # 'A thin single-line scratch defect on the PCB surface, narrow and sharp'
        # 'A wide linear scratch defect with base material exposed'
        # 'Multiple parallel scratch defects, aligned in one direction'
        # 'A large scratch defect covering a wide area'

    elif data == 'exp5':
        # Generate all 6 prompt combinations
        shade_list = ['dark', 'light']
        color_list = ['blue', 'gray', 'yellow']
        prompt_list = [f"A {shade} {color} dry film residual defect." for shade in shade_list for color in color_list]

        other_color_list = ['green', 'purple']
        other_prompt_list = [f"A {color} dry film residual defect." for color in other_color_list]

        prompt_list += other_prompt_list
    
    elif data == 'exp5_v2':
        # Generate all 6 prompt combinations
        shade_list = ['dark', 'light']
        color_list = ['blue', 'gray', 'yellow', 'purple']
        prompt_list = [f"A {shade} {color} dry film residual defect." for shade in shade_list for color in color_list]

        other_color_list = ['green'] # 'transparent'
        other_prompt_list = [f"A {color} dry film residual defect." for color in other_color_list]

        prompt_list += other_prompt_list

    elif data == 'exp6':
        defect_list = ['A crash', 'A scratch', 'White contamination', 'Red contamination', 'Foreign matter']
        prompt_list = [f"{defect} defect on tray." for defect in defect_list]
    
    # Repeat the 8 prompts evenly across the batch
    repeat_factor = batch_size // len(prompt_list)
    prompt_batch = prompt_list * repeat_factor

    # If not perfectly divisible, pad the remainder
    remainder = batch_size % len(prompt_list)
    if remainder > 0:
        prompt_batch += prompt_list[:remainder]

    return prompt_batch

def set_prompts_given(batch_size, prompt):
    prompt_list = [f"{prompt}"] * batch_size

    # Repeat the 8 prompts evenly across the batch
    repeat_factor = batch_size // len(prompt_list)
    prompt_batch = prompt_list * repeat_factor

    # If not perfectly divisible, pad the remainder
    remainder = batch_size % len(prompt_list)
    if remainder > 0:
        prompt_batch += prompt_list[:remainder]

    return prompt_batch