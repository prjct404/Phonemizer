# from openai import OpenAI
# import json

# def get_response(messages):
#     client = OpenAI(api_key = "", base_url="https://api.metisai.ir/api/v1/wrapper/deepseek")
#     response = client.chat.completions.create(model="deepseek-chat", messages=messages, max_tokens=100)
#     response_json = response.json()

#     return response.choices[0].message.content
os.environ["OPENROUTER_API_KEY"]="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"

from openai import OpenAI
from GE2PE import GE2PE
with open(r"prompt_base.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
g2p = GE2PE(model_path='homo-t5') 
base_prompt= ""
for line in lines:
    base_prompt+=line.strip()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

import requests
import json

def get_response(messages):
    url = "https://api.metisai.ir/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "tpsg-ua0Nt6z9KmotY9kXx1rsnjbrj45vvbY"  # Insert API key after Bearer
    }
    data = {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": messages,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    
    return response_json['choices'][0]['message']['content']


def openai_api():
    
    with open(r"prompt_base.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    g2p = GE2PE(model_path='homo-t5') 
    base_prompt= ""
    for line in lines:
        base_prompt+=line.strip()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

def phonemize(text):
    model_output = g2p.generate(text)
    completion = client.chat.completions.create(
    
        model="google/gemini-2.5-flash",
        messages=[
            {
                "role": "system",
                "content": base_prompt
            },
            {
                "role": "user",
                "content": f"{model_output}"
            }
        ]
    )
    
    return completion.choices[0].message.content





def main():
#metis test
#     message= [
#     {
#       "role": "system",
#       "content": "You are a Persian phonology-to-script reconstruction expert.\nYour task is to convert phonemic Pinglish text (Latinized phonemes, with special symbols) into fully phonemized Persian script with diacritics.\n\n### Core Rules:\n1. Always output in Persian script (UTF-8).\n2. Apply all diacritics:\n   - َ (fatha), ِ (kasra), ُ (damma)\n   - ْ (sukun) for consonants without vowels\n   - ّ (shadda) for doubled consonants\n   - Maddah (آ) where required\n3. Reconstruct correct Persian orthography, not Arabic (use گ, پ, ژ, ک, ی where needed).\n4. Never translate or paraphrase. Only reconstruct script with proper diacritics.\n5. Apply phoneme diacritics to **all words**. Do not leave any word or syllable without vowels or diacritics.\n6. If the model output contains ambiguous phonemes, choose the most standard/likely Persian letter unless clear from context.\n7. Correct common dictation falsings (e.g., 'tArikh' → تاريخ, not تارِيخ).\n8. Use dictionary overrides for high-frequency words of Arabic origin (e.g., احترام, اعمال, قرآن, ظلم, طلا).\n\n### Special Symbol Mappings (Model Output → Persian):\n- S  → ش\n- C  → چ\n- ?a → اَ\n- ?o → اُ\n- ?e → اِ\n- ?i → ای (with kasra)\n- Aa → آ (maddah)\n- g  → گ\n- j  → ج\n- q  → ق\n- x  → خ\n- zh → ژ\n- gh → غ\n- ph → ف\n- th → ث\n- dh → ذ\n- sh → ش (same as S)\n- ch → چ (same as C)\n\n### Handling Ambiguities:\n- For 'z' → default to ز unless dictionary context suggests ذ, ض, or ظ.\n- For 's' → default to س unless dictionary context suggests ث or ص.\n- For 't' → default to ت unless dictionary context suggests ط.\n- For 'h' → distinguish ه vs ح based on known words.\n- For 'a' → short vowel (َ) unless prefixed with ?a (then اَ).\n- For long vowels:\n  - aa → آ or اَ depending on context\n  - oo → اُ / و\n  - ee → اِ / ی\n\n### Examples:\n- Input: ketAb   → Output: كِتاب\n- Input: SahrA   → Output: شَهراء\n- Input: CAdor   → Output: چادُر\n- Input: ?aSAn   → Output: اَسان\n- Input: mo?o    → Output: مُؤُ\n- Input: zaban   → Output: زَبان\n- Input: ghorbat → Output: غُربَت\n- Input: tArikh  → Output: تاريخ (with ط)\n- Input: ?ehterAm → Output: اِحتِرام\n- Input: ?a?mAl  → Output: اِعمال"
#     },
#     {
#       "role": "user",
#       "content": "xAneye tang ?amma garm bud"
#     }
#   ]

#     x=get_response(message)
#     print(json.dumps(x, indent=2, ensure_ascii=False))

#openAI
    test_input = "در این بازی لم خاصی وجود دارد."
    output_test = phonemize(test_input)
    print(output_test)


   

if __name__ == "__main__":
    main()

