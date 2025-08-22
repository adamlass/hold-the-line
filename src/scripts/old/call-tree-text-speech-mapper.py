import json

TREES_LOCATION = "data/trees"
TREE_NAME = "tree1"
call_tree_path = f"{TREES_LOCATION}/{TREE_NAME}/callTreeReplaced.json"
callTree = json.load(open(call_tree_path, "r"))

mapped_call_tree = {}

def map_text_to_speech(tree: dict):
    """
    Map text to speech for each node in the call tree.
    """
    for content in tree["contents"]:
        if "text" in content:
            text = content["text"]
            file_id = content["fileID"]
            mapped_call_tree[file_id] = {
                "text": text,
                "file_path": f"speech/{file_id}.wav"
            }
            
            if "nextSegment" in content:
                next_segment_content = content["nextSegment"]
                if next_segment_content is not None:
                    map_text_to_speech(next_segment_content)
        
        next_segment_segment = tree["nextSegment"]
        if next_segment_segment is not None:
            map_text_to_speech(next_segment_segment)
            
    return callTree


mapped = map_text_to_speech(callTree)

# save the mapped call tree to a file
mapped_call_tree_path = f"{TREES_LOCATION}/{TREE_NAME}/mappedCallTree.json"
json.dump(mapped_call_tree, open(mapped_call_tree_path, "w"), indent=4)