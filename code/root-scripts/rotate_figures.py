
import re
import sys

log_file = 'rotation_log.txt'
file_path = '0-0-0-gravity-submission-aaa.tex'

with open(log_file, 'w') as log:
    log.write("Script started.\n")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        log.write(f"Read {len(content)} bytes.\n")
        
        regex = r'\\includegraphics\[width=[^\]]+\]\{[^}]+\}'
        matches = re.findall(regex, content)
        log.write(f"Found {len(matches)} matches.\n")
        
        def replace_func(match):
            full_match = match.group(0)
            if 'angle=-90' in full_match:
                return full_match
            new_match = full_match.replace('\\includegraphics[width=', '\\includegraphics[angle=-90,width=')
            log.write(f"Replacing: {full_match} -> {new_match}\n")
            return new_match

        new_content = re.sub(regex, replace_func, content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            log.write("File updated successfully.\n")
        else:
            log.write("No changes needed.\n")
            
    except Exception as e:
        log.write(f"Error: {str(e)}\n")

