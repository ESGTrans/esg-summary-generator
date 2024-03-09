from langchain_core.prompts import PromptTemplate

prompt_template = """請依據下文撰寫具體簡明的摘要:


"{text}"


具體簡明的摘要:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
