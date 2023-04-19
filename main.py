from Comparator import Comparator
from Variator import Variator

if __name__ == '__main__':
    question = "What is the capital of Ukraine?"
    answer = "The capital of Ukraine is Kyiv."

    variator = Variator()
    header, variations = variator.get_prompt_variations(question)
    print(f"Prompt variations: {variations}")

    comparator = Comparator()
    similarity_df = comparator.compute_similarity_for_variations([question] + variations)
    print("Similarity DataFrame:")
    print(similarity_df)
