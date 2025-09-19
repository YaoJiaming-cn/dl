tup3 = "a", "b", "c", "d"
print(type(tup3))

tup1 = (50)
tup2 = (50,)
print(type(tup1))
print(type(tup2))

tup1 = ('Google', 'Runoob', 1997, 2000)
tup2 = (1, 2, 3, 4, 5, 6, 7)

print("tup1[0]: ", tup1[0])
print("tup2[1:5]: ", tup2[1:5])

#元组中的值不可以修改

# 以下修改元组元素操作是非法的。
# tup1[0] = 100

tup4 = (12, 34.56)
tup5 = ('abc', 'xyz')

tup6 = tup4 + tup5
print (tup6)

#整个元组可以被删除
tup = ('Google', 'Runoob', 1997, 2000)

print(tup)
del tup
print("删除后的元组 tup : ")
# print(tup)


print("---------")
print(len((1, 2, 3)))
print(('Hi!',) * 4) # 复制
print(3 in (1, 2, 3)) #布尔

# 迭代
for x in (1, 2, 3):
    print (x, end=" ")

tup = ('Google', 'Runoob', 'Taobao', 'Wiki', 'Weibo','Weixin')
print(tup[1:])

tuple2 = ('5', '4', '8')
print(max(tuple2), min(tuple2))

# 将可迭代系列转换为元组
list1= ['Google', 'Taobao', 'Runoob', 'Baidu']
tuple1=tuple(list1)
print(tuple1)

# id可以查看内存地址
# 重新赋值的元组 tup，绑定到新的对象了，不是修改了原来的对象
tup = ('r', 'u', 'n', 'o', 'o', 'b')
print(id(tup))
tup = (1,2,3)
print(id(tup))

