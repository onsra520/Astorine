import torch.nn as nn
from torch.utils.data import Dataset

class NeuralNetwork(nn.Module):
    """
    Neural Network Model đơn giản bằng cách sử dụng thư viện PyTorch.
        - torch: Thư viện chính của PyTorch, cung cấp các công cụ để làm việc với tensor và tính toán trên GPU.
        - torch.nn: Module chứa các lớp và hàm để xây dựng mạng nơ-ron.
        - nn.Module: Lớp cơ sở cho tất cả các mô hình trong PyTorch.

    Args:
        input_size: Kích thước đầu vào của mô hình (số lượng đặc trưng).
        hidden_size: Kích thước của lớp ẩn (số lượng nơ-ron trong lớp ẩn).
        num_classes: Kích thước đầu ra của mô hình (số lượng lớp cần phân loại).
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__() # Gọi hàm khởi tạo của lớp cha (nn.Module).
        self.l1 = nn.Linear(input_size, hidden_size) # Lớp đầu tiên, ánh xạ từ input_size đến hidden_size.
        self.l2 = nn.Linear(hidden_size, hidden_size) # Lớp thứ hai, ánh xạ từ hidden_size đến hidden_size.
        self.l3 = nn.Linear(hidden_size, num_classes) # Lớp thứ ba, ánh xạ từ hidden_size đến num_classes.
        self.relu = nn.ReLU() # Hàm kích hoạt ReLU (Rectified Linear Unit), được sử dụng để thêm tính phi tuyến vào mô hình.
    
    def forward(self, x): # Hàm forward, xây dựng kiến trúc mạng nơ-ron.
        """
        Trong bài toán phân loại, hàm kích hoạt Softmax thường được sử dụng ở lớp cuối cùng để chuyển đổi 
        đầu ra thành xác suất. Tuy nhiên, trong PyTorch, hàm CrossEntropyLoss (thường được sử dụng cho bài 
        toán phân loại) đã bao gồm Softmax bên trong nó. Do đó, không cần áp dụng Softmax ở lớp cuối cùng.
        """        
        out = self.l1(x) # Ánh xạ đầu vào x qua lớp đầu tiên (l1)
        out = self.relu(out) # Áp dụng hàm kích hoạt ReLU cho đầu ra của lớp đầu tiên.
        out = self.l2(out) # Ánh xạ đầu ra của lớp thứ nhất qua lớp thứ hai (l2).
        out = self.relu(out) # Áp dụng hàm kích hoạt ReLU cho đầu ra của lớp thứ hai.
        out = self.l3(out) # Ánh xạ đầu ra của lớp thứ hai qua lớp thứ ba (l3).
        return out

class ChatDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    # Hỗ trợ việc truy cập bằng chỉ số để có thể lấy mẫu thứ i trong tập dữ liệu bằng cách dùng dataset[i].
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Chúng ta có thể gọi len(dataset) để trả về kích thước.
    def __len__(self):
        return self.n_samples

