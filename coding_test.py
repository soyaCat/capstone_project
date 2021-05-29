myarr = []
for i in range(ord('z')-ord('a')+1):
    myarr.append(0)

my_input = input()
my_input = list(my_input)
for index, i in enumerate(my_input):
    if ord('A')<=ord(i) and ord('Z')>=ord(i):
        i = chr(ord(i)+(ord('a')-ord('A')))
    my_input[index] = i

for i in my_input:
    myarr[ord(i)-ord('a')]+=1

max_num = max(myarr)
max_index = -1
for i in range(len(myarr)):
    if max_index == -1:
        if myarr[i] == max_num:
            max_index = i
    else:
        if myarr[i] == max_num:
            max_index = -2

if max_index == -2:
    print('?')
else:
    print(chr((max_index)+ord('A')))
