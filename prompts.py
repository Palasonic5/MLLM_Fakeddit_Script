"""
Zero-shot prompt templates for Fakeddit evaluation.
Each model's eval script imports from here so prompts stay consistent.
"""

SYSTEM_2WAY = (
    "You are a fake news detector. Given a Reddit post's title and image, "
    "determine whether the post is real or fake news."
)

USER_2WAY = (
    'Post title: "{title}"\n\n'
    "Look at the image and read the title carefully. "
    "Is this post real or fake news?\n"
    'Answer with ONLY one word: "real" or "fake".'
)

SYSTEM_6WAY = (
    "You are a fake news detector. Given a Reddit post's title and image, "
    "classify it into one of six misinformation categories."
)

USER_6WAY = (
    'Post title: "{title}"\n\n'
    "Look at the image and read the title carefully. "
    "Classify this post into one of the following categories:\n"
    "- true: genuine, accurate news\n"
    "- satire: satirical or parody content\n"
    "- misleading: misleading content or false context\n"
    "- imposter: content from imposter news sources\n"
    "- false: false connection between image and title\n"
    "- manipulated: manipulated or fabricated visual content\n\n"
    "Answer with ONLY one word from: true, satire, misleading, imposter, false, manipulated."
)


def get_prompts(label_type="2_way"):
    if label_type == "2_way":
        return SYSTEM_2WAY, USER_2WAY
    return SYSTEM_6WAY, USER_6WAY


# Response text → integer label
RESPONSE_MAP_2WAY = {"real": 0, "fake": 1}
RESPONSE_MAP_6WAY = {
    "true": 0,
    "satire": 1,
    "misleading": 2,
    "imposter": 3,
    "false": 4,
    "manipulated": 5,
}


def parse_response(text, label_type="2_way"):
    """
    Map model's text response to integer label.
    Returns None if response cannot be parsed.
    """
    mapping = RESPONSE_MAP_2WAY if label_type == "2_way" else RESPONSE_MAP_6WAY
    cleaned = text.strip().lower().rstrip(".,!?\"'")
    for key, val in mapping.items():
        if key in cleaned:
            return val
    return None
