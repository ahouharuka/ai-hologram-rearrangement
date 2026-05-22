# name = input('name: ')
# gender = input('gender: ')
# car_num = []
# i = 1
# while True:
#     car_num.append(input(f'car{i}\'s number: '))
#     i += 1




a = {
    'name': '张三',
    'gender': 'm',
    'car_num': [112,134]
}

b = {
    'name': '李四',
    'gender': 'f',
    'car_num': [101]
}

people = {'zhangsan': a, 'lisi': b}

t = input()

print((people.get(t)))

# print(people[t]['gender'])


