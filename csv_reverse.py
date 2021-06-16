f = open('data/test1.csv', 'r')
content = []
while True:
    buf = f.readline()
    if buf == '':
        break
    content.append(buf)
f.close()
f = open('data/test1.csv', 'w')
for i in range(1, len(content) + 1):
    f.write(content[-i])
f.close()