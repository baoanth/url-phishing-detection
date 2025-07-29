import pandas as pd

def preprocess_file(filename):
    # Đọc file không có header, gán tên cột là label và url
    df = pd.read_csv(filename, header=None, names=['label', 'url'], on_bad_lines='skip')
    
    # Thay giá trị label == 2 thành 0
    df['label'] = df['label'].replace(2, 'good')
    df['label'] = df['label'].replace(1, 'bad')

    # Ghi lại vào chính file cũ, không ghi index, dùng encoding utf-8
    df.to_csv(filename, index=False, encoding='utf-8')

if __name__ == "__main__":
    preprocess_file('train.csv')
    preprocess_file('test.csv')
    #preprocess_file('val.csv')
    print("✅ Đã xử lý xong và lưu lại các file.")
