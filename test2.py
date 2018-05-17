# @File  : test2.py
# @Author: 沈昌力
# @Date  : 2018/5/15
# @Desc  :
dict = {}
dict['123'] = 111
dict['234'] = 222
print(dict)

print(dict.get('234'))
dict['567'] = 333
dict.pop('123')
print(dict)