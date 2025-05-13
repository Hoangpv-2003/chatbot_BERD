import random
import json
import torch
import mysql.connector
from transformers import BertTokenizerFast, BertForTokenClassification
import google.generativeai as genai
from transformers import PhobertTokenizer, RobertaForTokenClassification
from sentence_transformers import SentenceTransformer, util
# Cấu hình Gemini API (thay bằng API key của bạn)
import sys
sys.stdout.reconfigure(encoding='utf-8')

genai.configure(api_key="AIzaSyDRWI9dzYTkIGONTaM98vXg2HfxdQUMqVQ")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents từ file JSON
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Load dữ liệu mô hình
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]


bot_name = "Sam"

# Kết nối CSDL
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="05052003",
        database="myshop"
    )

# Tải mô hình BERT NER
ner_model = RobertaForTokenClassification.from_pretrained('./bert_ner_model', local_files_only=True).to(device)
tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base', local_files_only=True)
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Mỗi pattern sẽ lưu (tag, pattern, embedding)
intent_embeddings = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        embedding = embedder.encode(pattern, convert_to_tensor=True)
        intent_embeddings.append({
            "tag": intent["tag"],
            "pattern": pattern,
            "embedding": embedding,
            "responses": intent["responses"]
        })

# Hàm tìm tên sản phẩm từ câu sử dụng mô hình BERT NER
def extract_product_name(sentence):
    keywords = [
    'smart watch',
    'iphone 15',
    'galaxy watch',
    'xiaomi band',
    'samsung',
    'smart phone',
    'mens trimmer',
    'slr camera',
    'mixer grinder',
    'phone gimbal',
    'tablet keyboard',
    'wireless earbuds',
    'party speakers',
    'slow juicer',
    'wireless headphones'
]

    for keyword in keywords:
        if keyword.lower() in sentence.lower():
            return keyword
    # Tokenize input sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # Get the model's output
    with torch.no_grad():
        outputs = ner_model(**inputs)
        logits = outputs.logits

    # Get the predictions (labels)
    predictions = torch.argmax(logits, dim=2)
    
    # Convert input IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = predictions[0].cpu().numpy()

    # Define the label list
    label_list = ["O", "B-PRODUCT", "I-PRODUCT"]
    
    product_name = []
    is_in_product = False

    for token, label_id in zip(tokens, predicted_labels):
        # Skip CLS and SEP tokens
        if token == tokenizer.cls_token or token == tokenizer.sep_token:
            continue

        # Debug output
        print(f"Token: {token}, Label: {label_list[label_id]}") 
        
        if label_list[label_id] == "B-PRODUCT":
            is_in_product = True  # Start of a product name
            product_name.append(token)
        elif label_list[label_id] == "I-PRODUCT" and is_in_product:
            # If it's part of the product name, continue appending
            product_name.append(token)
        else:
            is_in_product = False  # No longer a product

    # Clean up the token list for subwords, removing '##' prefix
    product_name = [token[2:] if token.startswith("##") else token for token in product_name]

    # Return the product name as a string
    return ' '.join(product_name) if product_name else None



# Kiểm tra sản phẩm còn hàng
def check_product_in_db(product_name):
    db = connect_db()
    cursor = db.cursor()
    query = "SELECT * FROM product WHERE LOWER(title) LIKE %s AND inStock > 0"
    like_pattern = f"%{product_name.lower()}%"
    cursor.execute(query, (like_pattern,))
    result = cursor.fetchone()
    cursor.close()
    db.close()

    if result:
        product_title = result[2]  # cột title
        return f"Dạ, bên mình có sản phẩm '{product_title}' còn hàng nhé!"
    else:
        return f"Xin lỗi, hiện tại bên mình chưa có sản phẩm '{product_name}'."

# Lấy giá sản phẩm
def get_product_price(product_name):
    db = connect_db()
    cursor = db.cursor()
    query = "SELECT title, price FROM product WHERE LOWER(title) LIKE %s"
    like_pattern = f"%{product_name.lower()}%"
    cursor.execute(query, (like_pattern,))
    result = cursor.fetchone()
    cursor.close()
    db.close()

    if result:
        product_title, price = result
        return f"Giá của sản phẩm '{product_title}' hiện tại là {price:,}00000 VND."
    else:
        return f"Mình chưa tìm thấy giá của sản phẩm '{product_name}' nhé!"

# Trả lời tự nhiên bằng Gemini nếu model cũ không chắc chắn
def gemini_response(user_input):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(user_input)
        if response and hasattr(response, 'text'):
            return response.text.strip()
        else:
            return None
    except Exception as e:
        print("Lỗi trong gemini_response:", e)
        return None

# Hàm trả lời chính
'''
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "check_product":
                    product_name = extract_product_name(msg)
                    if product_name:
                        return check_product_in_db(product_name)
                    else:
                        return "Bạn vui lòng cho biết tên sản phẩm cụ thể nhé!"
                
                elif tag == "check_price":
                    product_name = extract_product_name(msg)
                    if product_name:
                        return get_product_price(product_name)
                    else:
                        return "Bạn vui lòng cho biết tên sản phẩm để mình báo giá nhé!"
                
                else:
                    return random.choice(intent['responses'])

    # Nếu model không chắc chắn → Thử dùng Gemini trả lời
    try:
        gemini_reply = gemini_response(msg)
        if gemini_reply:
            return gemini_reply
    except Exception as e:
        print("Lỗi Gemini API:", e)

    return "Xin lỗi, mình chưa hiểu ý bạn nói. Bạn có thể nói lại được không?"
'''
def get_response(msg):
    # Trích xuất tên sản phẩm bằng BERT
    product_name = extract_product_name(msg)

    # Encode câu hỏi người dùng
    query_embedding = embedder.encode(msg, convert_to_tensor=True)

    # Tính cosine similarity với từng pattern đã encode
    max_score = -1
    best_match = None
    for item in intent_embeddings:
        score = util.cos_sim(query_embedding, item["embedding"]).item()
        if score > max_score:
            max_score = score
            best_match = item

    # Nếu điểm đủ cao (ví dụ > 0.65), xử lý theo intent
    if best_match and max_score > 0.8:
        tag = best_match["tag"]
        responses = best_match["responses"]

        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "check_product":
                    product_name = extract_product_name(msg)
                    if product_name:
                        return check_product_in_db(product_name)
                    else:
                        return "Bạn vui lòng cho biết tên sản phẩm cụ thể nhé!"
                
                elif tag == "check_price":
                    product_name = extract_product_name(msg)
                    if product_name:
                        return get_product_price(product_name)
                    else:
                        return "Bạn vui lòng cho biết tên sản phẩm để mình báo giá nhé!"
                
                else:
                    return random.choice(intent['responses'])
    # Nếu không khớp → fallback sang Gemini
    try:
        gemini_reply = gemini_response(msg)
        if gemini_reply:
            return gemini_reply
    except Exception as e:
        print("Lỗi gọi Gemini:", e)

    return "Xin lỗi, mình chưa hiểu ý bạn nói. Bạn có thể nói lại được không?"

# Khởi động chatbot
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        resp = get_response(sentence)
        print(resp)


