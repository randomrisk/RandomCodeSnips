import json

# Read JSON file
with open('E:\miccai_2023_papers.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)

# extract informations
extracted_info = []
for article in articles:
    extracted_info.append({
        "Title": article['Title'],
        "Abstract": article['Abstract'],
        "Topics": article.get('Topics', 'N/A')  # 
    })

# write
with open('E:\miccai_2023_abs.json', 'w', encoding='utf-8') as f:
    json.dump(extracted_info, f, ensure_ascii=False, indent=4)

print("Finsihed")
