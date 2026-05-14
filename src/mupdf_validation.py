import pymupdf4llm  

source = './data/salatiel_classes/Idolatria.pdf'
mk_response = pymupdf4llm.to_markdown(source)

with open('./data/markdown_export/idolatria.md', 'w', encoding='utf-8') as file:
    file.write(mk_response)

print("Markdown export completed successfully.")