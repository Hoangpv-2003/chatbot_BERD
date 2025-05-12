import random
import json
import torch
import mysql.connector
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from underthesea import pos_tag
from transformers import BertTokenizerFast, BertForTokenClassification
import google.generativeai as genai

# Cấu hình Gemini API (thay bằng API key của bạn)
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

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

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
ner_model = BertForTokenClassification.from_pretrained('./bert_ner_model', local_files_only=True).to(device)
tokenizer = BertTokenizerFast.from_pretrained('./bert_ner_model', local_files_only=True)

# Hàm tìm tên sản phẩm từ câu sử dụng mô hình BERT NER
def extract_product_name(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = ner_model(**inputs)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = predictions[0].cpu().numpy()

    label_list = ["O", "B-PROD", "I-PROD"]
    
    product_name = []
    for token, label_id in zip(tokens, predicted_labels):
        print(f"Token: {token}, Label: {label_list[label_id]}")  # Debug output
        if label_list[label_id] == "B-PROD" or label_list[label_id] == "I-PROD":
            if token.startswith("##"):
                token = token[2:]  # Loại bỏ dấu '##' ở các subword
            product_name.append(token)

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
        return f"Xin lỗi, hiện tại bên em chưa có sản phẩm '{product_name}'."

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
        return f"Giá của sản phẩm '{product_title}' hiện tại là {price:,} VND."
    else:
        return f"Mình chưa tìm thấy giá của sản phẩm '{product_name}' nhé!"

# Trả lời tự nhiên bằng Gemini nếu model cũ không chắc chắn
def gemini_response(user_input):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(user_input)
    return response.text.strip()

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
    product_name = extract_product_name(msg)
    if product_name:
        response = f"Dạ vâng, chúng tôi có bán {product_name}. Bạn muốn tìm hiểu thêm chứ?"
    else:
        response = "Bạn vui lòng cho tôi biết rõ tên sản phẩm bạn đang quan tâm nhé?"
    return response
# Khởi động chatbot
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        resp = get_response(sentence)
        print(resp)
