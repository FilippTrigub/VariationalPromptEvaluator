import os

import pandas as pd
import nltk
from dotenv import load_dotenv
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Variator import OpenAIQuery


class Comparator:
    """
    Class to compare the similarity between LLM I/O.
    """

    def __init__(self):
        # Download the necessary NLTK packages
        nltk.download('punkt')
        nltk.download('wordnet')
        load_dotenv()
        self.openai_query = OpenAIQuery(api_key=os.getenv('OPENAI_API_KEY'))

    def compute_similarity_for_variations(self, input_variations: list) -> DataFrame:
        """
        Calculates the similarity between the a list of similar questions and their answers using NLP techniques.
        :param question:
        :param variations:
        :return:
        """
        variations = pd.DataFrame(columns=['prompt', 'completion', 'score'])
        for variation in input_variations:
            completion = self.openai_query.completions(prompt_text=variation, n=1, return_prompt=False)[0]['text']
            variations = variations._append(
                {'prompt': variation, 'completion': completion,
                 'score': self.compute_similarity(variation, completion)}, ignore_index=True)
        variations = variations.sort_values(by=['score'], ascending=False)
        return variations

    @staticmethod
    def compute_similarity(question: str, answer: str) -> float:
        """
        Calculates the similarity between the question and answer using NLP techniques

        Args:
            question (str): prompt provided to LLM
            answer (str): answer generated by LLM

        Returns:
            score (float): the similarity score between 0 and 1 assigned to the answer
        """
        # Preprocess the text
        corpus = [question] + [answer]

        # Compute the similarity score using cosine similarity metric
        vect = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf = vect.fit_transform(corpus)
        similarity_score = cosine_similarity(tfidf)

        return similarity_score[0][1]