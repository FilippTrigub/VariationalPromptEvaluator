# Variational Prompt Evaluator
This repository contains a simple tool to evaluate the effects of small variations of prompts on the responses.
The tool instructs GPT to generate variations without changing the meaning, thus allowing the LLm itself to define interpret the accepted level of variation.
Finally, the tool produces responses for each variation including the initial one and ranks them using the cosine similarity.

Using this approach it is possible to quantize the precision of the LLM under the assumption of insignificant fluctuations in responses to identical inputs. 