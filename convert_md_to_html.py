import markdown2

with open('/home/lr-2002/project/reasoning_manipulation/ManiSkill/docs/source/tasks/coin_bench/index.md', 'r') as md_file:
    md_content = md_file.read()

html_content = markdown2.markdown(md_content, extras=['tables'])

with open('/home/lr-2002/project/reasoning_manipulation/ManiSkill/github_page/index.html', 'w') as html_file:
    html_file.write('<html><head><title>COIN Bench Tasks</title></head><body>')
    html_file.write(html_content)
    html_file.write('</body></html>')

print('Conversion completed successfully.')
