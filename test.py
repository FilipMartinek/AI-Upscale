arr = range(100)
temp = len(arr) // 2
i = temp
x = 58

while not arr[i] == x:
    print(i, " ", temp)

    temp = temp // 2
    if temp == 0: temp = 1

    if arr[i] > x:
        i -= temp
    else:
        i += temp
print(f"answer: {i}")