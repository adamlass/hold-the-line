import copy
from dataclasses import asdict
import json
import sys
import time
from classes import Content, DialOption, Segment, SegmentType, Output, OutputField, GoalsOutput
import numpy as np
from tqdm import tqdm
from utils import generate_unique_id
import os
from dotenv import load_dotenv
from scripts.OpenAIService import OpenAIService

load_dotenv()
service = OpenAIService()

PROCESSED_DATA_FOLDER = "data_processed"
TREES_PATH = f"{PROCESSED_DATA_FOLDER}/trees"
DESCRIPTION_FILE_NAME = "description.txt"
CALL_TREE_FILE_NAME = "callTree.json"

MAX_TRIES = 3
RETRY_WAIT_TIME = 10

SYSTEM_PROMPT = '''
You are a helpful and precise assistant.
Given a description of a company. Fill out the marked text fields of a fictive IVR call menu.
I will use your output directly to generate fictive IVR call audio with TTS, 
therefore, please make sure that the text is speakable.

Please be creative based on the company description and make sure that the text is speakable and makes sense in relation to the fictive company.
If it is a dial option, make sure to mention that they should press '[INDEX]' to choose that menu.
I will replace it with the correct index in the output, so you are only allowed to write '[INDEX]' in the text.
A dial option can never be an offer to call back, returning to the menu or visit a website, it should always be a menu option leading to a new segment.

I have already created a json structure containing the marked text fields that you should fill out in the output. 
This is marked with 'ROOT:<reference_index>'.
{call_tree}

The welcome intro should be one sentence only!
The number of references in the output should match the number of references in the input.
It's very important that you fill out all the references in the output!!!
'''

USER_PROMPT = '''
Here is the company information that the output should be specific to:
--------{company_description}
--------
'''

cached_prompts = {}

def generateSegment(max_width, max_depth, parent_reference=None) -> Segment:
    add_depth: bool = np.random.choice([True]) # TODO make this a parameter
    if max_depth <= 0 or not add_depth:
        return Segment(type=SegmentType.holdTheLine, contents=[Content(text=f"{parent_reference}:0", fileID=generate_unique_id())], id=generate_unique_id())
    else:
        return generateDialOptions(max_width, max_depth, parent_reference)

def generateDialOptions(max_width, max_depth, parent_reference=None) -> Segment:
    width = np.random.randint(2, max_width + 1)
    dial_options = []
    for i in range(1, width+1):
        reference = f"{parent_reference}:{i}"
        dialOption = DialOption(text=reference, index=i, nextSegment=generateSegment(max_width, max_depth - 1, reference), fileID=generate_unique_id())
        dial_options.append(dialOption)
        
    return Segment(type=SegmentType.dialOptions, contents=dial_options, id=generate_unique_id())

def generateCallTree(max_width, max_depth) -> Segment:
    root_reference = "ROOT"
    segment = Segment(type=SegmentType.welcome, contents=[Content(text=root_reference, fileID=generate_unique_id())], nextSegment=generateDialOptions(max_width, max_depth, root_reference), id=generate_unique_id())
    return segment

def get_reference_text(reference, reference_map: dict):
    result = reference_map.get(reference)
    assert result is not None, f"Reference {reference} not found in output"
    return result

def replace_references(segment: Segment, reference_map: dict) -> Segment:
    new_contents = []
    for content in segment.contents:
        content_text:str = get_reference_text(content.text, reference_map)
        if hasattr(content, "index"):
            content_text = content_text.replace("[INDEX]", str(content.index))
            new_contents.append(DialOption(index=content.index, text=content_text, nextSegment=replace_references(content.nextSegment, reference_map), fileID=content.fileID))
        else:
            new_contents.append(Content(text=content_text, fileID=content.fileID))
            
    next_segment = None
    if segment.nextSegment is not None:
        next_segment = replace_references(segment.nextSegment, reference_map)
        
    new_segment: Segment = Segment(type=segment.type, contents=new_contents, nextSegment=next_segment, id=segment.id)
    return new_segment

def get_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def __replace_ids_with_null(segment: dict):
    """
    Replaces all IDs in the segment with None.
    """
    del segment["id"]
    del segment["navigationGoals"]
    for content in segment["contents"]:
        del content["fileID"]
        if segment["type"] == SegmentType.dialOptions:
            __replace_ids_with_null(content["nextSegment"])
    
    if segment["nextSegment"] is None:
        del segment["nextSegment"]
    else:
        __replace_ids_with_null(segment["nextSegment"])
        
def replace_ids_with_null(segment: Segment) -> Segment:
    segment_copy: dict = asdict(segment)
    __replace_ids_with_null(segment_copy)
    return segment_copy

# get list of directories in TREES_PATH
directories = get_directories(TREES_PATH)
directories.sort(key=lambda x: int(x.split("_")[1]))
print(f"Directories in {TREES_PATH}: {len(directories)}")

for directory in tqdm(directories, leave=False):
    print("-"*40)
    call_tree_replaced_path = f"{TREES_PATH}/{directory}/callTreeReplaced.json"
    print(f"Checking if call tree already generated for {directory}")
    if os.path.exists(call_tree_replaced_path):
        print(f"Call tree already generated, skipping...")
        continue
    
    description_path = f"{TREES_PATH}/{directory}/{DESCRIPTION_FILE_NAME}"
    print(f"Loading description from {description_path}")
    with open(description_path, "r") as f:
        description = f.read()
    print(f"Loaded description: {description}")
    
    print(f"Generating call tree for {directory}")
    call_tree = generateCallTree(2, 2)
    print(f"Generated call tree")
    
    call_tree_path = f"{TREES_PATH}/{directory}/{CALL_TREE_FILE_NAME}"
    print(f"Saving call tree to {call_tree_path}")
    with open(call_tree_path, "w") as f:
        json.dump(asdict(call_tree), f, indent=4)
    print(f"Call tree saved to {call_tree_path}")
    
    call_tree_null_ids = replace_ids_with_null(call_tree)
    tree_hash = hash(json.dumps(call_tree_null_ids, indent=4))
    print(f"Call tree hash: {tree_hash}")
    if tree_hash not in cached_prompts:
        print("Generating new system prompt")
        tree_system_prompt = SYSTEM_PROMPT.format(call_tree=call_tree_null_ids)
        cached_prompts[tree_hash] = tree_system_prompt
    else:
        print("Using cached system prompt")
    
    cached_system_prompt = cached_prompts[tree_hash]
    user_prompt = USER_PROMPT.format(company_description=description)
    
    tries = 0
    while True:
        tries += 1
        try:
            print(f"Sending request to OpenAI API, try {tries}/{MAX_TRIES}")
            response: Output = service.parse_information(cached_system_prompt, user_prompt, Output)
            replacements_path = f"{TREES_PATH}/{directory}/replacements.json"
            print(f"Saving replacements to {replacements_path}")
            with open(replacements_path, "w") as f:
                json.dump(response.model_dump(), f, indent=4)
            print(f"Replacements saved to {replacements_path}")
            
            print("Replacing references in call tree using replacements")
            reference_map = {}
            for field in response.fields:
                reference_map[field.reference] = field.text
            call_tree_replaced = replace_references(call_tree, reference_map)
            print("Successfully replaced references in call tree")
            
            print(f"Saving replaced call tree to {call_tree_replaced_path}")
            with open(call_tree_replaced_path, "w") as f:
                json.dump(asdict(call_tree_replaced), f, indent=4)
            print(f"Replaced call tree saved to {call_tree_replaced_path}")
            
            break
        except Exception as e:
            print("Error:", e)
            if tries >= MAX_TRIES:
                print("Max tries reached, exiting...")
                sys.exit(1)
            time.sleep(RETRY_WAIT_TIME)
            print("Retrying...")
            
    time.sleep(0.1)
    
    