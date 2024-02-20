import re
from typing import Any, Dict, List, Text
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

from rasa.nlu.constants import TOKENS_NAMES, MESSAGE_ATTRIBUTES
from underthesea import word_tokenize

class VietnameseTokenizer(Tokenizer):

    provides = [TOKENS_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super().__init__(component_config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)
        words = word_tokenize(text)
        print(text)
        return self._convert_words_to_tokens(words, text)

    def _convert_words_to_tokens(self, words: List[Text], text: Text) -> List[Token]:
        tokens = []
        running_offset = 0

        for word in words:
            try:
                word_offset = text.index(word, running_offset)
                word_len = len(word)
                running_offset = word_offset + word_len

                tokens.append(Token(
                    word,
                    start=word_offset,
                    end=running_offset,
                    lemma=word,  # You may need to replace this with the actual lemma
                ))
            except ValueError:
                # Handle the case where the word is not found in the text
                # You can choose to skip this word or handle it differently
                pass

        return tokens
