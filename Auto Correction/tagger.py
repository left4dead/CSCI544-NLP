__author__ = 'arpit'

import os
import subprocess
import sys
import nltk

megaMPath = '/home/arpit/MegaM/megam_0.92/megam'


def main():
    testfile = ''

    argsLength = len(sys.argv)
    if 1 <= len(sys.argv) < 2:
        print 'Please enter command line arguments as below:'
        print 'python tagger.py <Test File>'
        exit()
    if len(sys.argv) == 2:
        testfile = str(sys.argv[1])
    if len(sys.argv) > 2:
        print 'Please enter exactly one argument to run this script!!!'
        exit()

    TagAndGenerate(testfile)


def TagAndGenerate(testfile):
    testFileData = ''
    inputData = ''

    if os.path.isfile(testfile):
        with open(testfile, 'r') as currentfile:
            for line in currentfile:
                if line == '\n':
                    testFileData += line
                    continue

                inputData += line.strip()
                text = inputData.split()
                tagged = nltk.pos_tag(text)

                inputData = ''
                for (word, tag) in tagged:
                    inputData += word + '/' + tag + ' '

                testFileData += inputData.strip() + '\n'
                inputData = ''
    else:
        print 'Please enter a valid test file path!!!'
        exit()

    output = ''
    for line in testFileData.splitlines():
        lineData = line.split()

        for i in range(len(lineData)):
            if len(lineData[i]) > 0:
                model = ''
                data = lineData[i].rsplit('/', 1)

                if len(data) != 2:
                    print 'The parsed data is not in the format Word/POSTag. ' \
                          'Below is the first occurrence of this data as \'index : actual data\'\n'
                    print str(i) + ' ' + str(data) + '\n'
                    exit()

                temp = data[0]
                data[0] = data[0].lower()

                if data[0] == 'too' or data[0] == 'to':
                    model = 'train.pos.too.model'
                if data[0] == 'loose' or data[0] == 'lose':
                    model = 'train.pos.loose.model'
                if data[0] == 'it\'s' or data[0] == 'its':
                    model = 'train.pos.its.model'
                if data[0] == 'you\'re' or data[0] == 'your':
                    model = 'train.pos.your.model'
                if data[0] == 'they\'re' or data[0] == 'their':
                    model = 'train.pos.their.model'

                if model == '':
                    output += temp + ' '
                    continue

                prevTag = ''
                nextTag = ''

                if i >= 1:
                    prevTag += lineData[i-1].rsplit('/', 1)[1]
                if i+1 < len(lineData):
                    nextTag += lineData[i+1].rsplit('/', 1)[1]

                testFileData = data[0] + ' pwt:' + prevTag + ' nwt:' + nextTag

                f = open(testfile + '.megam', 'w')
                f.write(testFileData)
                f.close()

                pred = predictTags(model, testfile + '.megam')
                word = pred.split()[0]

                if temp[:1].isupper():
                    if temp.isupper():
                        word = pred.split()[0].capitalize()
                    else:
                        word = temp[:1] + pred.split()[0][1:]

                output += word + ' '

                os.remove(testfile + '.megam')

        output = output + '\n'

    f = open(testfile + '.out', 'w')
    f.write(output)
    f.close()


def predictTags(modelfile, testFile):
    p = subprocess.Popen([megaMPath, '-predict', os.getcwd() + '/' + modelfile, '-nc', 'multiclass', testFile],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    (output, err) = p.communicate()
    return output


main()