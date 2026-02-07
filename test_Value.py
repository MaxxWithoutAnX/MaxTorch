from Value import Value

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
    prod._backward()
    assert val2.grad == 4*1