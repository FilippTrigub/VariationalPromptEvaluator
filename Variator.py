import os
from typing import Tuple, List, Any

from dotenv import load_dotenv
import openai
import re


class OpenAIQuery:
    def __init__(self, api_key, max_length: int = 2000, stop_sequence: str = None, temperature: float = 0.5):
        """
        Constructor for the OpenAIQuery class.

        Initializes class variables.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.max_length = max_length
        self.stop_sequence = stop_sequence
        self.temperature = temperature

    def completions(self, prompt_text, n=1, presence_penalty=0.0, frequency_penalty=0.0, best_of=1,
                    return_prompt=False):
        """
        Generates completions for the given prompt.

        n: Number of completions to generate.
        presence_penalty: Float value controlling the penalty applied to each potential completion based on how
                          similar it is to previous items in the result list.
        frequency_penalty: Float value controlling the penalty applied to each item based on its frequency in the
                           training dataset.
        best_of: Integer value controlling the number of output sequences to generate during beam search.
        return_prompt: Boolean indicating whether or not to include the prompt in the returned completions.
        """
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_text,
            temperature=self.temperature,
            max_tokens=self.max_length,
            n=n,
            stop=self.stop_sequence,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            echo=return_prompt,
        )

        completions = []
        for choice in response.choices:
            completion = {}
            completion['id'] = choice.index
            completion['text'] = re.sub(r'[^\x00-\x7F]+', ' ', choice.text)
            completions.append(completion)
        return completions

    def set_max_length(self, length):
        """
        Sets the maximum length of generated completions.
        """
        self.max_length = length

    def set_temperature(self, temp):
        """
        Sets the temperature value for the generation process.
        """
        self.temperature = temp

    def set_stop_sequence(self, stop_sequence):
        """
        Sets the presence or absence of a stop sequence for completions.
        """
        self.stop_sequence = stop_sequence

    def get_settings(self):
        """
        Returns the current settings of the OpenAIQuery class.
        """
        settings = {}
        settings['prompt'] = self.prompt_text
        settings['max_length'] = self.max_length
        settings['temperature'] = self.temperature
        settings['stop_sequence'] = self.stop_sequence
        return settings


class Variator:
    """
    Class to generate prompt variations for the LLM I/O.
    """

    def __init__(self):
        # Load the config file
        load_dotenv()
        self.openai_query = OpenAIQuery(api_key=os.getenv('OPENAI_API_KEY'))

    def get_prompt_variations(self, prompt):
        """
        Returns a list of prompt variations to be used in the LLM I/O
        Args:
            prompt (str): prompt to be used in the LLM I/O

        Returns:
            prompt_variations (list): list of prompt variations
        """
        variator_prompt = f'You are a master Linguist and writer with a creative mind. ' \
                          f'Provide 10 variations of the following phrase without changing its meaning.' \
                          f'{prompt}'
        response = self.openai_query.completions(prompt_text=variator_prompt, n=1, return_prompt=True)[0]['text']
        return self.split_enumeration(response)

    @staticmethod
    def split_enumeration(enumeration_string: str) -> tuple[str, list[Any]]:
        """
        Splits an enumeration into a list containing the header and each item.
        """
        header_regex = re.compile(r'^(.*?)\n(?:\-|\+|\*|\d\.)', flags=re.MULTILINE | re.DOTALL)
        match = header_regex.search(enumeration_string)

        header = match.group(1).strip() if match else ""
        items = [x.strip()[3:] for x in re.findall(r'(?:^\-|\+|\*|\d\..+?$)', enumeration_string, flags=re.MULTILINE)]

        return header, items
