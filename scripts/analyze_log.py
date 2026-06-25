import re
import json

with open('logs/autonomous_trader/1m_primary_run_v12.log', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
content = ansi_escape.sub('', content)

out = []
lines = content.split('\n')
for i, line in enumerate(lines):
    if '18:46:15' in line and any(x in line for x in ['LLM Decision', 'Regime filter', 'position_opened', 'BUY', 'pipeline']):
        start = max(0, i-3)
        end = min(len(lines), i+4)
        for j in range(start, end):
            out.append(lines[j])
        out.append('---')

with open('logs/analysis_1846.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Wrote to logs/analysis_1846.txt')
