class BaseData:
    def __init__(self,name):
        self.name = name
        self.nameList = []
        self.dictionary = {}

    def getName(self):
        return self.name
    def getAllName(self):
        return self.nameList

    def getByName(self,name):
        return self.dictionary[name]

    def getLabelByIndex(self,index):
        return self.nameList[index]
	
    def appendNameList(self,name):
        self.nameList.append(name);	

    def appendDictionary(self,name,value):		
        self.dictionary[name]=value

    def updateDictionary(self,row):
        for i in range(len(row)):
            self.dictionary[self.nameList[i]].append(row[i])

class starRead(BaseData):	
    def __init__(self,name):
        BaseData.__init__(self,name)		
        self.filePointer = open(name)	
        self.readFile()

    def readFile(self):
        readlineFlag = True
        while True:
            if readlineFlag:
                line = self.filePointer.readline()
            else:
                readlineFlag = True
			
            if len(line) is 0:
                break

            line = line.rstrip();
            #if we read an empty line
            if len(line) is 0:
                continue

            if line.startswith('data_'):
                #here we start a new table
                self.appendNameList(line)
                table = BaseData(line)
                tableName = line;
                #consider two type: with or without loop_				
                line = self.filePointer.readline();					
                while not (line.startswith('_rln') or line.startswith('loop_')):#eliminate empty lines
                    line = self.filePointer.readline().rstrip()
				
                if line.startswith('loop_'):
                    #with loop_
                    while not line.startswith('_rln'):
                        line = self.filePointer.readline().rstrip() #eliminate empty lines

                    readlineFlag = False
		    while True:
                        if readlineFlag:
                            line = self.filePointer.readline()						
                        else:
                            readlineFlag = True					
                            #a new block start or we meet the end of line
                        if len(line) is 0 or line.startswith('data_'):
                            readlineFlag = False
                            break
                            # there may be emptyline 
                        if len(line.rstrip()) is 0:
                            continue
                        if line.startswith('_rln'):
                            lineList = line.split();			
                            table.appendNameList(lineList[0])
                            table.appendDictionary(lineList[0],[])
                        else:							
                            #update a new row
                            table.updateDictionary(line.split())
                else:
                    #without loop_
                    readlineFlag = False
                    while True:
                        if readlineFlag:
                            line = self.filePointer.readline();							
                        else:
                            readlineFlag = True
                        #end of file or a new data block
                        if line is '' or line.startswith('data_'):								
                            readlineFlag = False
                            break
                        # there may be emptyline 
                        if line.rstrip() is '':
                            continue
                        if line.startswith('_rln'):
                            lineList = line.split();
                            table.appendNameList(lineList[0])
                            table.appendDictionary(lineList[0],lineList[1])
                self.appendDictionary(tableName,table)
