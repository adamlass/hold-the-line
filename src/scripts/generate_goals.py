

import json
import os
import sys
import time

from classes import Content, DialOption, Segment, SegmentType, Output, OutputField, GoalsOutput
from tqdm import tqdm
from dotenv import load_dotenv
from scripts.OpenAIService import OpenAIService

load_dotenv()
service = OpenAIService()

PROCESSED_DATA_FOLDER = "data_processed"
TREES_PATH = f"{PROCESSED_DATA_FOLDER}/trees"
CALL_TREE_REPLACED_FILE_NAME = "callTreeReplacedWithGoals.json"
N_PERSONAS = 5
MAX_TRIES = 3
SKIP_GENERATED = True

__PERSONAS = [
    "A busy professional who values efficiency and quick resolutions.",
    "An ex-military veteran seeking support.",
    "A tech-savvy individual who likes to use technical terms.",
    "An elderly person who may need extra assistance.",
    "A well-formulated student of physics.",
    "A first-time caller unfamiliar with the company's services.",
    "A dyslexic mother of two young children.",
    "A small business owner with limited funds.",
    "A customer with a specific complaint or issue.",
    "A refugee from Ukraine who speaks English as a second language."
]
PERSONAS_USED = __PERSONAS[:N_PERSONAS]


GOAL_PROMPT = '''
I'm training an LLM agent to navigate a fictive IVR call menu. However, this specific "LLM agent" is not you.
You are a helpful and precise assistant that helps me generate fake user goals that the "LLM agent" should base its navigation on.

A real scenario use of the service I'm building will be as follows:
1. The user picks up a phone and opens the Hold The Line app.
2. The users dials a number. A chat box opens.
3. The "LLM agent" ask the user what they need help with.
4. The user respond with their intended goal.

Your job is, based on the provided persona description, to generate realistic user goals that could be used in this scenario.
The input sequences corresponds to the relevant information that the "LLM agent" would use to navigate based on the goal that you generate.

Rules:
- The goals should be specific to the unique segment sequences.
- Two goals should never overlap. The goals should be as orthogonal to each other as possible.
- The goals should always be from the perspective of the user of the Hold The Line app. Short, simple and realistic.
- The users goal formulation should be based on the persona description.
- Always generate a unique goal for each unique segment sequence in the input.
- Try to avoid using the exact formulations of the input in the goals. However, this might not always be possible.
- The output should be a list of unique goals where each goal corresponds to the unique segment sequence in the input.
'''

GOAL_USER_PROMPT = '''
Here is the persona description of the user that you should imitate:
"{persona}"

Here is the list of unique sequences that you should generate goals for:
{unique_sequences}
'''

# get unique routes
def get_unique_segment_sequences(segment: Segment) -> list[list[str]]:
    result = []
    
    if segment["type"] == SegmentType.holdTheLine:
        return [[content["text"] for content in segment["contents"]] + [segment["id"]]]
    elif segment["type"] == SegmentType.dialOptions:
        for content in segment["contents"]:
            sub_sequences = get_unique_segment_sequences(content["nextSegment"])
            for sub_sequence in sub_sequences:
                result.append([content["text"]] + sub_sequence)
    elif segment["type"] == SegmentType.welcome:
        sub_sequences = get_unique_segment_sequences(segment["nextSegment"])
        for sub_sequence in sub_sequences:
            result.append([content["text"] for content in segment["contents"]] + sub_sequence)
        
    return result

def get_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# get list of directories in TREES_PATH
directories = get_directories(TREES_PATH)
directories.sort(key=lambda x: int(x.split("_")[1]))
print(f"Directories in {TREES_PATH}: {len(directories)}")

def set_goals_in_call_tree(segment: Segment, goals_map: dict) -> Segment:
    if segment["id"] in goals_map:
        if segment["navigationGoals"] is None:
            segment["navigationGoals"] = []
        segment["navigationGoals"].extend(goals_map[segment["id"]])
    
    for content in segment["contents"]:
        if segment["type"] == SegmentType.dialOptions:
            set_goals_in_call_tree(content["nextSegment"], goals_map)
            
    if segment["nextSegment"] is not None:
        set_goals_in_call_tree(segment["nextSegment"], goals_map)

for directory in tqdm(directories):
    print("-"*40)
    call_tree_replaced_with_goals_path = f"{TREES_PATH}/{directory}/{CALL_TREE_REPLACED_FILE_NAME}"
    print(f"Checking if replaced call tree exists at {directory}")
    if SKIP_GENERATED and os.path.exists(call_tree_replaced_with_goals_path):
        print(f"Replaced call tree already exists, skipping...")
        continue
    
    call_tree_replaced_path = f"{TREES_PATH}/{directory}/callTreeReplaced.json"
    
    print(f"Loading replaced call tree from {call_tree_replaced_path}")
    with open(call_tree_replaced_path, "r") as f:
        call_tree_replaced: Segment = json.load(f)
    
    unique_sequences = get_unique_segment_sequences(call_tree_replaced)
    unique_sequences_map = {}
    for sequence in unique_sequences:
        unique_sequences_map[sequence[-1]] = sequence[:-1]
    unique_sequence_keys = list(unique_sequences_map.keys())
    unique_sequence_values = list(unique_sequences_map.values())
    
    sequence_goal_map = {key:[] for key in unique_sequence_keys}
    
    for persona in PERSONAS_USED:
        user_prompt = GOAL_USER_PROMPT.format(
            persona=persona,
            unique_sequences=unique_sequence_values
        )
        tries = 0
        while True:
            tries += 1
            try:
                print(f"Generating goals for persona: {persona}")
                goal_response: GoalsOutput = service.parse_information(GOAL_PROMPT, user_prompt, GoalsOutput)
                assert len(goal_response.goals) == len(unique_sequences_map), f"Goal response length {len(goal_response.goals)} does not match unique sequences length {len(unique_sequences_map)}"
                
                for i, goal in enumerate(goal_response.goals):
                    sequence_key = unique_sequence_keys[i]
                    sequence_goal_map[sequence_key].append(goal)
                
                break
            except Exception as e:
                print(f"Error: {e}")
                if tries >= MAX_TRIES:
                    print("Max tries reached, stopping...")
                    sys.exit(1)
                print("Retrying...")
                time.sleep(1)
                continue
        
    print(f"Setting goals in call tree")
    set_goals_in_call_tree(call_tree_replaced, sequence_goal_map)
    
    print(f"Saving call tree to {call_tree_replaced_with_goals_path}")
    with open(call_tree_replaced_with_goals_path, "w") as f:
        json.dump(call_tree_replaced, f, indent=4)
    print(f"Call tree saved to {call_tree_replaced_with_goals_path}")
    
    time.sleep(0.1)