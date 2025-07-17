import numpy as np

# تعريف دالة التفعيل tanh
def tanh(x):
    return np.tanh(x)

# إعداد البيانات
# مدخلات (x1, x2)
x = np.array([0.6, -0.2])  # مثال يمكن تغييره

# الأوزان من المدخلات إلى الطبقة المخفية (2 neurons في الطبقة المخفية)
W_hidden = np.random.uniform(-0.5, 0.5, (2, 2))  # 2 hidden neurons × 2 inputs
b1 = 0.5

# حساب مخرجات الطبقة المخفية
z_hidden = np.dot(W_hidden, x) + b1
a_hidden = tanh(z_hidden)

# الأوزان من الطبقة المخفية إلى الطبقة النهائية
W_output = np.random.uniform(-0.5, 0.5, (1, 2))  # 1 output neuron × 2 hidden neurons
b2 = 0.7

# حساب مخرجات الطبقة النهائية
z_output = np.dot(W_output, a_hidden) + b2
a_output = tanh(z_output)

# طباعة النتائج
print("Input: ", x)
print("Hidden Layer Weights:\n", W_hidden)
print("Hidden Layer Output (after tanh):", a_hidden)
print("Output Layer Weights:\n", W_output)
print("Final Output of the Network (after tanh):", a_output)
