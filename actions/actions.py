# This file contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these actions:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted
import requests


class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Get user message from Rasa tracker
        user_message = tracker.latest_message.get('text')
        print(user_message)

        # Make API call to OpenAI's GPT-3.5-turbo
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': 'Bearer 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',
            'Content-Type': 'application/json'
        }
        data = {
            'model': "gpt-3.5-turbo",
            'messages': [
                {'role': 'system', 'content': 'You are an AI assistant for the user. You help to solve user query'},
                {'role': 'user', 'content': 'You: ' + user_message}
            ],
            'max_tokens': 100
        }

        # Make the API request
        response = requests.post(url, headers=headers, json=data)

        # Process the response
        if response.status_code == 200:
            chatgpt_response = response.json()
            message = chatgpt_response['choices'][0]['message']['content']
            dispatcher.utter_message(message)
        else:
            # Handle error
            return [f"Sorry, I couldn't generate a response. HTTP Status Code: {response.status_code}"]

        # Revert user message which led to fallback.
        return [UserUtteranceReverted()]
