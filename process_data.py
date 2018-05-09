f = open('data/ucf101/flow.txt', 'r')
f1 = open('flow.txt', 'w')
for line in f.readlines():
    
    f1.write(line.replace('/data4/zhouhao/dataset/ucf101/tvl1_flow/{:s}/', ''))
f.close()
f1.close()
