class MyOpenFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.f_handle = None
    
    def __enter__(self):
        print("enter...", self.filename)
        self.f_handle = open(self.filename)
        return self.f_handle
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit...")
        if self.f_handle:
            self.f_handle.close()
        if exc_type is not None:
            print("exception:", exc_type, exc_val, exc_tb)
            return False
        return True

with MyOpenFile("./rl/rl_test.py") as f:
    for line in f:
        print(line)
    # raise ZeroDivisionError
# 装饰器contextmanager。
# 该装饰器将一个函数中yield语句之前的代码当做__enter__方法执行，
# yield语句之后的代码当做__exit__方法执行。
# 同时yield返回值赋值给as后的变量

import contextlib
@contextlib.contextmanager
def open_func(file_name):
    # __enter__方法
    print('open file:', file_name, 'in __enter__')
    file_handler = open(file_name, 'r')

    yield file_handler

    # __exit__方法
    print('close file:', file_name, 'in __exit__')
    file_handler.close()
    return
print('------------------')
with open_func("./rl/rl_test.py") as file_in:
    for line in file_in:
        print(line)