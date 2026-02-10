from Value import Value
import math

def test_data():
    val = Value(8)
    assert (val.data == 8)

def test_add():
    val1 = Value(8)
    val2 = Value(3)
    sum = val1+val2
    assert sum.data == 11

def test_add_non_val():
    val1 = Value(8)
    not_val = 3
    sum = val1 + not_val
    assert sum.data == 11, "Can not add two Value"
    assert isinstance(sum, Value), "Sum is not a Value"

def test_add_int_first():
    non_val = 4
    val = Value(7)
    sum = non_val + val
    assert sum.data == 11, "can not add non val first"

def test_mul():
    val1 = Value(6)
    val2 = Value(4)
    prod = val1*val2
    assert prod.data == 24

def test_mul_non_val():
    val1 = Value(6)
    val2 = 4
    prod = val1*val2
    assert prod.data == 24

def test_rmul():
    val1 = Value(6)
    val2 = 4
    prod = val2*val1
    assert prod.data == 24

def test_backward_single():
    val1 = Value(4)
    val2 = Value(5)
    prod = val1*val2
    prod.backward()
    assert val2.grad == 4*1

def test_sub():
    val1 = Value(4)
    val2 = Value(5)
    sub = val1 - val2
    assert sub.data == -1

def test_sigmoid():
    val1 = Value(4)
    sig_val = val1.sigmoid()
    sig_manual = 1/(1+math.exp(-4))
    assert sig_val.data == sig_manual

def test_sigmoid_backprop():
    val1 = Value(4)
    sig_val = val1.sigmoid()
    sig_manual = 1/(1+math.exp(-4))
    sig_val.backward()
    sig_grad = sig_manual * (1-sig_manual)
    assert val1.grad == sig_grad

def test_tanh():
    val1 = Value(4)
    tanh_val = val1.tanh()
    sig_manual = (math.exp(2*4)-1)/(math.exp(2*4)+1)
    assert tanh_val.data == sig_manual

def test_tanh_backprop():
    val1 = Value(4)
    tanh_val = val1.tanh()
    tanh_manual = (math.exp(2*4)-1)/(math.exp(2*4)+1)
    tanh_val.backward()
    tanh_grad = 1 - tanh_manual**2
    assert val1.grad == tanh_grad

def test_relu():
    val1 = Value(4)
    relu_val = val1.ReLU()
    relu_val.backward()
    assert relu_val.data == 4
    assert val1.grad == 1

def test_negative_ReLU():
    val1 = Value(-4)
    relu_val = val1.ReLU()
    relu_val.backward()
    print(relu_val, val1)
    assert relu_val.data == 0
    assert val1.grad == 0