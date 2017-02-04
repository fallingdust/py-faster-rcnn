#!/usr/bin/env python

with open('train.txt.bak', 'r') as f:
    line = f.readline()
    while line:
        parts = line.split(' ')
        filename = parts[0].replace('images', 'annotations').replace('png', 'txt')
        with open(filename, 'w') as af:
            count = int(parts[1])
            for i in range(0, count):
                a = []
                a.append('foreground')
                a.extend(parts[i*4+2:(i+1)*4+2])
                line = ' '.join(a)
                af.write(line)
                if i < count - 1:
                    af.write('\n')
        line = f.readline()
