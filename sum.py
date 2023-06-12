from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

import unittest


class PaperSummarizer:

    def __init__(self):
        """Initializes the PaperSummarizer class"""
        pass

    def preprocess_text(self, text_list):
        """
        Method for preprocessing the input text.

        Parameters:
        text (str): the text to be preprocessed

        Returns:
        str: the preprocessed text
        """
        prompt = '''
        使用中文摘要它，以下内容：
        {text}
        '''
        PROMPT = PromptTemplate(template=prompt, input_variables=["text"])
        texts = [Document(page_content=text) for text in text_list]
        llm = OpenAI(
            # model='gpt-3.5-turbo',
            temperature=0, 
            max_tokens=2000
        )
        chain = load_summarize_chain(
            llm, 
            chain_type="map_reduce",
            map_prompt=PROMPT,
            combine_prompt=PROMPT,
            verbose=True,
        )
        response = chain.run(texts)
        return response

    def generate_summary(self, text):
        """
        Method for generating the summary of the input text.

        Parameters:
        text (str): the text to be summarized

        Returns:
        str: the summarized text
        """
        pass

    def postprocess_summary(self, summary):
        """
        Method for postprocessing the generated summary.

        Parameters:
        summary (str): the generated summary

        Returns:
        str: the postprocessed summary
        """
        pass


def test():
    """Tests the PaperSummarizer class"""
    summarizer = PaperSummarizer()
    text_list = [
        '''
        Introduction: The COVID-19 pandemic highlighted the importance of making 
        epidemiological data and scientific insights easily accessible and explorable 
        for public health agencies, the general public, and researchers. 
        Stateof-the-art approaches for sharing data and insights included regularly 
        updated reports and web dashboards. However, they face a trade-off between 
        the simplicity and flexibility of data exploration. With the capabilities of 
        recent large language models (LLMs) such as GPT-4, this trade-off can be overcome.
        ''',
        '''
        Results: We developed the chatbot “GenSpectrum Chat” (cov-spectrum.org/chat) 
        which uses GPT-4 as the underlying large language model (LLM) to explore SARS-CoV-2 
        genomic sequencing data. Out of 500 inputs from real-world users, the chatbot 
        provided a correct answer for 453 prompts; an incorrect answer for 13 prompts, 
        and no answer although the question was within scope for 34 prompts. We also tested 
        the chatbot with inputs from 10 different languages, and despite being provided 
        solely with English instructions and examples, it successfully processed prompts 
        in all tested languages.
        ''',
        '''
        Conclusion: LLMs enable new ways of interacting with information systems. 
        In the field of public health, GenSpectrum Chat can facilitate the analysis 
        of real-time pathogen genomic data. With our chatbot supporting interactive 
        exploration in different languages, we envision quick and direct access 
        to the latest evidence for policymakers around the world.
        ''',
    ]
    response = summarizer.preprocess_text(text_list)
    print(response)

if __name__ == '__main__':
    test()