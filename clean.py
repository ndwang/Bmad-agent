import sys

def clean_latex_file(input_file, output_file):
    blank_line_before = True
    left_over_eliminate = ''  # To handle multiline constructs
    left_over_sub = ''        # To handle multiline constructs
    in_example = False
    
    def substitute(line, str1, sub=None):
        n1 = len(str1)
        has_subbed = False
        
        while True:
            ix = line.find(str1)
            if ix == -1:
                break
            has_subbed = True
            if sub is not None:
                line = line[:ix] + sub + line[ix+n1:]
            else:
                line = line[:ix] + line[ix+n1:]
        
        return line, has_subbed
    
    def eliminate2(line, str1, str2, sub1=None, sub2=None):
        n1 = len(str1)
        n2 = len(str2)
        ix1 = 0
        
        while True:
            # Find str1
            ix1 = line.find(str1, ix1)
            if ix1 == -1:
                return line, None, None
            if ix1 > 0 and line[ix1-1] == '\\':
                ix1 += 1
                continue
            
            # Find str2
            ix2 = line.find(str2, ix1 + n1)
            if ix2 == -1:
                # If ending string is not found then must be on a later line
                if sub1 is not None:
                    line = line[:ix1] + sub1 + line[ix1+n1:]
                else:
                    line = line[:ix1] + line[ix1+n1:]
                return line, str2, sub2
            
            # substitute
            if sub1 is not None:
                line = line[:ix1] + sub1 + line[ix1+n1:ix2] + (sub2 or '') + line[ix2+n2:]
                ix1 = ix1 + len(sub1) - 1
            else:
                line = line[:ix1] + line[ix1+n1:ix2] + line[ix2+n2:]
                ix1 = ix1 - 1
            
            if ix1 < 0:
                ix1 = 0
        
        return line, None, None
    
    def eliminate_inbetween(line, str1, str2, pad_with_blanks):
        n1 = len(str1)
        n2 = len(str2)
        blank = ' ' * 100
        
        while True:
            ix1 = line.find(str1)
            if ix1 == -1:
                return line
            ix2 = line.find(str2, ix1 + 1)
            if ix2 == -1:
                return line
            
            if pad_with_blanks:
                padding = blank[:ix2+n2-ix1]
                line = line[:ix1] + padding + line[ix2+n2:]
            else:
                line = line[:ix1] + line[ix2+n2:]
        
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.rstrip('\n')
            
            # Exit conditions
            if line.startswith('%%'):
                break
                
            if '\\begin{example}' in line:
                in_example = True
            if '\\end{example}' in line:
                in_example = False
                
            # Skip certain LaTeX commands
            if any(line.strip().startswith(cmd) for cmd in [
                '\\section', '\\subsection', '\\label', '\\begin', '\\end',
                '\\centering', '\\caption', '\\vskip']):
                continue
                
            if '\\index{' in line:
                continue

            # Skip lines with only '\newpage' or '--------'
            if line.strip() == '\\newpage' or line.strip().startswith('%---'):
                continue
                
            # Handle leftover eliminates from previous lines
            if left_over_eliminate != '':
                ix = line.find(left_over_eliminate.strip())
                if ix != -1:
                    line, _ = substitute(line, left_over_eliminate, left_over_sub)
                    left_over_eliminate = ''
                    left_over_sub = ''
            
            # Apply substitutions
            line, _ = substitute(line, "{}", "")
            line, new_left_over_eliminate, new_left_over_sub = eliminate2(line, '\\vn{', '}', '"', '"')
            if new_left_over_eliminate is not None:
                left_over_eliminate, left_over_sub = new_left_over_eliminate, new_left_over_sub
                
            line, new_left_over_eliminate, new_left_over_sub = eliminate2(line, '\\item[\\vn{\\{', '\\}}]', '   Argument: ', '')
            if new_left_over_eliminate is not None:
                left_over_eliminate, left_over_sub = new_left_over_eliminate, new_left_over_sub
                
            line, new_left_over_eliminate, new_left_over_sub = eliminate2(line, '\\item[', ']', '     ', '')
            if new_left_over_eliminate is not None:
                left_over_eliminate, left_over_sub = new_left_over_eliminate, new_left_over_sub
                
            line, _ = substitute(line, '\\item', '*')
            line, _ = substitute(line, '``\\vn', '"')
            line, _ = substitute(line, '``', '"')
            line, _ = substitute(line, '\\`', '`')
            line, _ = substitute(line, "}''", '"')
            line, _ = substitute(line, "''", '"')
            line, _ = substitute(line, '\\bf', '')
            line, _ = substitute(line, '\\arrowbf', '')
            line, has_subbed_value = substitute(line, '\\$', '$')
            
            if not has_subbed_value and not in_example:
                line, _ = substitute(line, '$')
                
            line, _ = substitute(line, '\\protect')
            line, _ = substitute(line, '\\_', '_')
            line, _ = substitute(line, '\\#', '#')
            line, _ = substitute(line, '\\%', '%')
            line, _ = substitute(line, '\\tao', 'Tao')
            line, _ = substitute(line, '\\bmad', 'Bmad')
            line = eliminate_inbetween(line, '& \\sref{', '}', True)
            line = eliminate_inbetween(line, '\\hspace*{', '}', True)
            line = eliminate_inbetween(line, '(\\sref{', '})', False)
            line = eliminate_inbetween(line, ' \\sref{', '}', False)
            line = eliminate_inbetween(line, '\\sref{', '}', False)
            line = eliminate_inbetween(line, '{\\it ', '}', False)
            line = eliminate_inbetween(line, '\\parbox{', '}', False)
            line, _ = substitute(line, '] \\Newline')
            line, _ = substitute(line, '\\Newline')
            
            if not in_example:
                line, _ = substitute(line, ' &')
                
            line, _ = substitute(line, '\\vfill')
            line, _ = substitute(line, '\\vfil')
            line, _ = substitute(line, '\\hfill')
            line, _ = substitute(line, '\\hfil')
            line, _ = substitute(line, '\\break')
            line, _ = substitute(line, '\\midrule')
            line, _ = substitute(line, '\\toprule')
            line, _ = substitute(line, '\\bottomrule')
            line, _ = substitute(line, '\\\\ \\hline')
            line, _ = substitute(line, '\\\\')
            line, _ = substitute(line, '\\W ', '^')
            line, _ = substitute(line, '"\\W"', '"^"')
            line, _ = substitute(line, '\\Bf ', '')
            line, _ = substitute(line, '\\B', '\\')
            line, _ = substitute(line, '\\chapter', 'chapter ')
            
            # Remove LaTeX comments
            if line.startswith('% '):
                line = line[2:]
            if line.startswith('%'):
                line = line[1:]
                
            # Remove leading braces
            while line and (line[0] == '{' or line[0] == '}'):
                line = line[1:]
                
            # Remove non-escaped braces
            i = 1
            while i < len(line):
                if (line[i] in ['{', '}']) and line[i-1] != '\\':
                    line = line[:i] + line[i+1:]
                else:
                    i += 1
                    
            line, _ = substitute(line, '\\{', '{')
            line, _ = substitute(line, '\\}', '}')
            line, _ = substitute(line, '\\(', '')
            line, _ = substitute(line, '\\)', '')
            
            # Remove trailing exclamation mark
            if line and line[-1] == '!':
                line = line[:-1] + ' '
                
            # Handle blank lines
            if line.strip() == '':
                if blank_line_before:
                    continue
                blank_line_before = True
            else:
                blank_line_before = False
                
            # Write the processed line
            f_out.write(line + '\n')

def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_latex.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        clean_latex_file(input_file, output_file)
        print(f"Successfully processed {input_file} and saved to {output_file}")
    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()