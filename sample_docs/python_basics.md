# Python Basics

Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with the use of significant indentation.

## Variables and Data Types

Python has several built-in data types:
- **int**: Integer numbers like 42, -7, 0
- **float**: Floating-point numbers like 3.14, -0.001
- **str**: Strings like "hello", 'world'
- **bool**: Boolean values True and False
- **list**: Ordered mutable collections like [1, 2, 3]
- **dict**: Key-value pairs like {"name": "Alice", "age": 30}
- **tuple**: Ordered immutable collections like (1, 2, 3)
- **set**: Unordered collections of unique elements like {1, 2, 3}

Variables in Python are dynamically typed. You do not need to declare their type:
```python
x = 42        # int
name = "Bob"  # str
pi = 3.14     # float
```

## Functions

Functions are defined using the `def` keyword:
```python
def greet(name):
    return f"Hello, {name}!"

result = greet("Alice")  # "Hello, Alice!"
```

Python supports default arguments, keyword arguments, and variable-length arguments:
```python
def power(base, exponent=2):
    return base ** exponent
```

## Control Flow

Python uses if/elif/else for conditional execution:
```python
if temperature > 30:
    print("It's hot!")
elif temperature > 20:
    print("It's nice.")
else:
    print("It's cold.")
```

Loops include for and while:
```python
for i in range(5):
    print(i)

while condition:
    do_something()
```

## List Comprehensions

Python supports concise list creation:
```python
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

## Error Handling

Python uses try/except for error handling:
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("This always runs.")
```
