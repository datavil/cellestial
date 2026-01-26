def my_function(a, b=2, *, c=3):
    pass

# Access positional default arguments
print(my_function.__defaults__)  # Output: (2,)

# Access keyword-only default arguments
print(my_function.__kwdefaults__)  # Output: {'c': 3}

dc = my_function.__kwdefaults__
dc.update({'c': 4})

print(dc)