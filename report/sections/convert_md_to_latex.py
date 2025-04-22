# convert_md_to_latex.py
import os
import re

def convert_md_to_latex(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert headings
    content = re.sub(r'^# (.*?)$', r'\\section{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*?)$', r'\\subsection{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.*?)$', r'\\subsubsection{\1}', content, flags=re.MULTILINE)
    
    # Convert bold and italic
    content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
    content = re.sub(r'\*(.*?)\*', r'\\textit{\1}', content)
    
    # Convert bullet lists
    content = re.sub(r'- (.*?)$', r'\\item \1', content, flags=re.MULTILINE)
    content = re.sub(r'(\\item [^\n]+)\n\n(?=\\item)', r'\1\n', content)
    content = re.sub(r'(\\item .+\n)(?=\\item|\n)', r'\1', content)
    lines = content.split('\n')
    list_started = False
    result = []
    
    for line in lines:
        if line.startswith('\\item') and not list_started:
            result.append('\\begin{itemize}')
            list_started = True
        elif not line.startswith('\\item') and list_started and line.strip():
            result.append('\\end{itemize}')
            list_started = False
        result.append(line)
        
    if list_started:
        result.append('\\end{itemize}')
        
    content = '\n'.join(result)
    
    # Convert figure references
    content = re.sub(r'!\[(.*?)\]\((.*?)\)', r'\\begin{figure}[!htb]\n\\centering\n\\includegraphics[width=0.8\\columnwidth]{\2}\n\\caption{\1}\n\\label{fig:\2}\n\\end{figure}', content)
    
    # Convert citations
    content = re.sub(r'\[@(.*?)\]', r'\\cite{\1}', content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

# Convert each section
sections = [
    'report/sections/1_introduction.md',
    'report/sections/2_background.md',
    'report/sections/3_methodology.md',
    'report/sections/4_results.md',
    'report/sections/5_challenges.md',
    'report/sections/6_conclusion.md'
]

os.makedirs('report/latex_sections', exist_ok=True)

for section in sections:
    base_name = os.path.basename(section)
    output_name = os.path.splitext(base_name)[0] + '.tex'
    output_path = os.path.join('report/latex_sections', output_name)
    convert_md_to_latex(section, output_path)
    print(f"Converted {section} to {output_path}")

print("All sections converted successfully!")