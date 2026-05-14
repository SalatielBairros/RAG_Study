import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from docling.document_converter import DocumentConverter  

source = './data/salatiel_classes/Idolatria.pdf'
converter = DocumentConverter() 
doc = converter.convert(source).document 
mk_response = doc.export_to_markdown()

with open('./data/markdown_export/idolatria.md', 'w', encoding='utf-8') as file:
    file.write(mk_response)

print("Markdown export completed successfully.")