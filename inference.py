# inference.py

from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# Load model and tokenizer
model = BertForTokenClassification.from_pretrained('./bert_ner_model', local_files_only=True)
tokenizer = BertTokenizerFast.from_pretrained("./bert_ner_model")

# Nhận input từ người dùng
sentence = "Tôi muốn mua iphone và laptop mới nhất."

# Tokenize input
inputs = tokenizer(sentence, return_tensors="pt")

# Dự đoán với model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Convert logits to predictions
predictions = torch.argmax(logits, dim=2)

# Lấy nhãn (labels) dự đoán
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = predictions[0].numpy()

# Gán tên nhãn (label) cho từng token
label_list = ["O", "B-PROD", "I-PROD"]

# Xử lý từ khóa như TV, iPhone, Laptop
product_keywords = ["tv", "iphone", "laptop", "máy tính", "điện thoại", "tivi"]

previous_token = None
current_product = False  # Biến để xác định nếu token thuộc về sản phẩm

for token, label_id in zip(tokens, predicted_labels):
    # Kiểm tra xem token có phải là subtoken (token bắt đầu bằng '##')
    if token.startswith('##'):
        token = previous_token  # Gán lại token trước đó (bỏ qua subtoken)
    else:
        previous_token = token  # Cập nhật token trước đó
    
    # Kiểm tra các từ khóa sản phẩm và gán nhãn B-PROD cho từ khóa
    if token.lower() in product_keywords:
        label_id = 1  # Gán nhãn 'B-PROD' cho từ khóa sản phẩm
        current_product = True  # Đánh dấu bắt đầu của sản phẩm
    elif current_product:
        label_id = 2  # Nếu đang trong chuỗi sản phẩm, gán nhãn 'I-PROD'
    
    print(f"{token}: {label_list[label_id]}")
    
    # Nếu token không phải là sản phẩm nữa, đặt lại trạng thái
    if label_id != 2:
        current_product = False
