def app(arg1, arg2):
    return arg1 + arg2  

def minus(a, b):
    return a - b

def divide(x, y):
    return x / y

def multiply(m, n):
    return m * n   

def fibonacci(n):
    if n <= 0:
        return "Input should be a positive integer"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b

if __name__ == '__main__':
    print(app(5, 3))            
    print(minus(10, 4))
    print(divide(20, 5))
    print(multiply(6, 7))
    print(fibonacci(10))
